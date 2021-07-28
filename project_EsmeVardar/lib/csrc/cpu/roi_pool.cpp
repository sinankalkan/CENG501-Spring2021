#include <float.h>
#include <torch/extension.h>

template <class T>
inline void add(T* address, const T& val) {
    *address += val;
}

template <typename T>
void roi_pool_forward(const T* input,
                      const T spatial_scale,
                      const int channels,
                      const int height,
                      const int width,
                      const int output_size,
                      const T* rois,
                      const int num_rois,
                      T* output,
                      int* argmax_data) {
    for (int n = 0; n < num_rois; ++n) {
        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];
        int roi_start_w = round(offset_rois[1] * spatial_scale);
        int roi_start_h = round(offset_rois[2] * spatial_scale);
        int roi_end_w = round(offset_rois[3] * spatial_scale);
        int roi_end_h = round(offset_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(output_size);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(output_size);

        for (int ph = 0; ph < output_size; ++ph) {
            for (int pw = 0; pw < output_size; ++pw) {
                int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
                int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
                int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
                int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

                // Add roi offsets and clip to input boundaries
                hstart = std::min(std::max(hstart + roi_start_h, 0), height);
                hend = std::min(std::max(hend + roi_start_h, 0), height);
                wstart = std::min(std::max(wstart + roi_start_w, 0), width);
                wend = std::min(std::max(wend + roi_start_w, 0), width);
                bool is_empty = (hend <= hstart) || (wend <= wstart);

                for (int c = 0; c < channels; ++c) {
                    // Define an empty pooling region to be zero
                    T maxval = is_empty ? 0 : -FLT_MAX;
                    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                    int maxidx = -1;

                    const T* input_offset = input + (roi_batch_ind * channels + c) * height * width;

                    for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                int input_index = h * width + w;
                                if (input_offset[input_index] > maxval) {
                                    maxval = input_offset[input_index];
                                    maxidx = input_index;
                            }
                        }
                    }
                    int index = ((n * channels + c) * output_size + ph) * output_size + pw;
                    output[index] = maxval;
                    argmax_data[index] = maxidx;
                } // channels
            } // output_size
        } // output_size
    } // num_rois
}

template <typename T>
void roi_pool_backward(const T* grad_output,
                       const int* argmax_data,
                       const int num_rois,
                       const int channels,
                       const int height,
                       const int width,
                       const int pooled_height,
                       const int pooled_width,
                       T* grad_input,
                       const T* rois,
                       const int n_stride,
                       const int c_stride,
                       const int h_stride,
                       const int w_stride) {    
    for (int n = 0; n < num_rois; ++n) {
        const T* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        for (int c = 0; c < channels; ++c) {
            T* grad_input_offset = grad_input + ((roi_batch_ind * channels + c) * height * width);
            const int* argmax_data_offset = argmax_data + (n * channels + c) * pooled_height * pooled_width;

            for (int ph = 0; ph < pooled_height; ++ph) {
                for (int pw = 0; pw < pooled_width; ++pw) {
                    int output_offset = n * n_stride + c * c_stride;
                    int argmax = argmax_data_offset[ph * pooled_width + pw];

                    if (argmax != -1) {
                        add(grad_input_offset + argmax,
                            static_cast<T>(grad_output[output_offset + ph * h_stride + pw * w_stride])
                        );
                    }
                } // pooled_width
            } // pooled_height
        } // channels
    } // num_rois
}

std::tuple<torch::Tensor, torch::Tensor> roi_pool_forward_cpu(const torch::Tensor& input,
                                                              const torch::Tensor& rois,
                                                              const float spatial_scale,
                                                              const int output_size) {
    AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
    AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");

    int num_rois = rois.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    const int pooling_width = output_size;
    const int pooling_height = output_size;

    torch::Tensor output = torch::zeros(
        {num_rois, channels, pooling_height, pooling_width}, input.options());
    torch::Tensor argmax = torch::zeros(
        {num_rois, channels, pooling_height, pooling_width},
        input.options().dtype(at::kInt));

    if (output.numel() == 0) {
        return std::make_tuple(output, argmax);
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "RoIPool_forward", [&] {
        roi_pool_forward<scalar_t>(
            input.contiguous().data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            output_size,
            rois.contiguous().data_ptr<scalar_t>(),
            num_rois,
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
    });

    return std::make_tuple(output, argmax);
}

torch::Tensor roi_pool_backward_cpu(const torch::Tensor& grad,
                                    const torch::Tensor& argmax,
                                    const torch::Tensor& input_size,
                                    const torch::Tensor& rois) {
    // Check if input tensors are CPU tensors
    AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");
    AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");
    AT_ASSERTM(argmax.device().is_cpu(), "argmax must be a CPU tensor");

    auto input_size_a = input_size.accessor<int,1>();
    const int batch_size = input_size_a[0];
    const int channels = input_size_a[1];
    const int height = input_size_a[2];
    const int width = input_size_a[3];

    const int num_rois = argmax.size(0);

    const int pooled_width = argmax.size(3);
    const int pooled_height = argmax.size(2);

    torch::Tensor grad_input = torch::zeros(
        {batch_size, channels, height, width}, grad.options());

    // handle possibly empty gradients
    if (grad.numel() == 0) {
        return grad_input;
    }

    // get stride values to ensure indexing into gradients is correct.
    int n_stride = grad.stride(0);
    int c_stride = grad.stride(1);
    int h_stride = grad.stride(2);
    int w_stride = grad.stride(3);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.scalar_type(), "RoIPool_backward", [&] {
        roi_pool_backward<scalar_t>(
            grad.data_ptr<scalar_t>(),
            argmax.data_ptr<int>(),
            num_rois,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois.contiguous().data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride);
    });

    return grad_input;
}

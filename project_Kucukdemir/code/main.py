import sys
import os
import signal
import time
from time import perf_counter

import numpy as np
import PIL.Image as im

import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms.functional as xform

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

class NoiseUtils:
    def apply_noise(img, noise_type, noise_std):
        # TODO(ff-k): we may add other types of distributions here, such as Poisson or Bernoulli
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_std, (img.size[1], img.size[0], len(img.getbands())))
        else:
            raise NotImplementedError('Unknown noise type: \'%s\'' % noise_type)
        
        return im.fromarray(np.uint8(np.clip((np.asarray(img, 'float') + (noise * 255.0)), 0, 255)))

class Noisier2NoiseDataset(data_utils.Dataset):
    def __init__(self, root_path, crop_dims, noise_type, noise_std, mode):
        super(Noisier2NoiseDataset, self).__init__()

        self.crop_dims = crop_dims
        self.noise_type = noise_type
        self.noise_std = noise_std

        self.root_path = root_path
        self.img_path = os.listdir(self.root_path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_path, self.img_path[index])
        img =  im.open(img_path).convert('RGB') # NOTE(ff-k): make sure the image contains 3 channels (i.e not grayscale)

        if img.size[0] < self.crop_dims[0] or img.size[1] < self.crop_dims[1]:
            scale_factor_x = self.crop_dims[0]/img.size[0]
            scale_factor_y = self.crop_dims[1]/img.size[1]
            scale_factor = max(scale_factor_x, scale_factor_y)
            scaled_w = int(round(img.size[0]*scale_factor))
            scaled_h = int(round(img.size[1]*scale_factor))
            img = img.resize((scaled_w, scaled_h), resample=im.BICUBIC)
        
        crop_tl_x = np.random.randint(0, img.size[0] - self.crop_dims[0] + 1)
        crop_tl_y = np.random.randint(0, img.size[1] - self.crop_dims[1] + 1)
        img = img.crop((crop_tl_x, crop_tl_y, crop_tl_x + self.crop_dims[0], crop_tl_y + self.crop_dims[1]))

        img_noisy = NoiseUtils.apply_noise(img, self.noise_type, self.noise_std)
        img_noisier = NoiseUtils.apply_noise(img_noisy, self.noise_type, self.noise_std)

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        tensor_noisy = xform.to_tensor(img_noisy).to(device=device, dtype=torch.float32)
        tensor_noisier = xform.to_tensor(img_noisier).to(device=device, dtype=torch.float32)

        return tensor_noisier, tensor_noisy

class UNet(nn.Module):
    def __init__(self, n, m):
        super(UNet, self).__init__()

        # NOTE(ff-k): As used in Noise2Noise, see Table 2

        leaky_relu_alpha = 0.1

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        self.encoder_0 = nn.Sequential(
                # ENC_CONV0
                nn.Conv2d(n, 48, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                )

        # NOTE(ff-k): where (1 <= i <= 5)
        self.encoder_i = nn.Sequential(
                # ENC_CONV_i
                nn.Conv2d(48, 48, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # ---------
                # POOL_i
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )

        self.encoder_6 = nn.Sequential(
                # ENC_CONV6
                nn.Conv2d(48, 48, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                )

        self.decoder_5 = nn.Sequential(
                # UPSAMPLE5
                nn.UpsamplingNearest2d(scale_factor=2),
                )

        self.decoder_4 = nn.Sequential(
                # DEC_CONV5A
                nn.Conv2d(96, 96, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # DEC_CONV5B
                nn.Conv2d(96, 96, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # UPSAMPLE4
                nn.UpsamplingNearest2d(scale_factor=2),
                )

        # NOTE(ff-k): where (3 >= i >= 1)
        self.decoder_i = nn.Sequential(
                # DEC_CONV_(i+1)_A
                nn.Conv2d(144, 96, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # DEC_CONV_(i+1)_B
                nn.Conv2d(96, 96, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # UPSAMPLE_i
                nn.UpsamplingNearest2d(scale_factor=2),
                )

        self.decoder_0 = nn.Sequential(
                # DEC_CONV1A
                nn.Conv2d(96+n, 64, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # DEC_CONV1B
                nn.Conv2d(64, 32, 3, stride=1, padding='same', device=device),
                nn.LeakyReLU(negative_slope=leaky_relu_alpha, inplace=True),
                # DEC_CONV1C
                nn.Conv2d(32, m, 3, stride=1, padding='same', device=device),
                )

    def forward(self, input):
        is_running = True
        while is_running == True:
            try:
                val = self.encoder_0(input)
                pool1 = self.encoder_i(val)
                pool2 = self.encoder_i(pool1)
                pool3 = self.encoder_i(pool2)
                pool4 = self.encoder_i(pool3)
                val = self.encoder_i(pool4)
                val = self.encoder_6(val)
                val = self.decoder_5(val)
                val = torch.cat((val, pool4), dim=1)
                val = self.decoder_4(val)
                val = torch.cat((val, pool3), dim=1)
                val = self.decoder_i(val)
                val = torch.cat((val, pool2), dim=1)
                val = self.decoder_i(val)
                val = torch.cat((val, pool1), dim=1)
                val = self.decoder_i(val)
                val = torch.cat((val, input), dim=1)
                val = self.decoder_0(val)
                is_running = False
            except:
                # NOTE(ff-k): try one more time after emptying the cache. it generally does not help, though
                torch.cuda.empty_cache()
                if is_running == False:
                    raise
                else:
                    is_running = False
        return val

def interrupt_handler(signum, frame):
    print('Received an interrupt signal and will terminate after completing the active epoch')
    globals()['interruped_by_user'] = True

def get_run_config(checkpoint=None):
    if checkpoint:
        run_config = checkpoint['run_config']
    else:
        run_config = {}
        run_config['dataset_id'] = 'coco'
        run_config['noise_type'] = 'gaussian'
        run_config['noise_std'] = 0.25

        # NOTE(ff-k): As used in Noise2Noise, see Section 3.1
        run_config['crop_width'] = 256
        run_config['crop_height'] = 256
        
        # NOTE(ff-k): In the original work, batch_size is 32 (Noisier2Noise, see Section 4).
        #             However, it is reduced to 16 in order to avoid 'out of memory' errors
        run_config['batch_size'] = 16

        # NOTE(ff-k): As used in Noisier2Noise, see Section 4
        run_config['adam_learning_rate_first_phase'] = 0.001
        run_config['adam_learning_rate_second_phase'] = 0.0001
        run_config['second_phase_start_progress'] = 0.9 # NOTE(ff-k): For example, if there are 165000 batches 
                                                        #            in total, we will jump to second phase 
                                                        #            after 150000 batches

        # NOTE(ff-k): As used in Noise2Noise, see Section A.2
        run_config['adam_beta_1'] = 0.9
        run_config['adam_beta_2'] = 0.99
        run_config['adam_eps'] = 1e-8

        # NOTE(ff-k): the number of epochs is not specified in none of the papers and our code does not 
        #             use a max_epoch value. instead, we will keep running until the user interrupts
        run_config['epoch'] = 0

        # NOTE(ff-k): number of input and output channels, respectively. we may need to change 
        #             number of input channels (i.e. 'n') in some cases (e.g. when we use Monte Carlo
        #             rendered images as input)
        run_config['n'] = 3
        run_config['m'] = 3
        
    return run_config

def main(mode, force_dataset, save_prob):

    checkpoints_path = '../checkpoints/'
    checkpoints = sorted(os.listdir(checkpoints_path))
    if len(checkpoints) > 0:
        print('Using latest checkpoint: %s' % (checkpoints[len(checkpoints)-1]))
        if mode == 'train':
            print('Make sure to clean checkpoints directory if you want to start training from scratch')
        checkpoint = torch.load(checkpoints_path + checkpoints[len(checkpoints)-1])
    else:
        checkpoint = {}

    cfg = get_run_config(checkpoint)
    if force_dataset != '':
        cfg['dataset_id'] = force_dataset

    print('Running | mode: %s, dataset: %s' % (mode, cfg['dataset_id']));

    pytorch_model = UNet(cfg['n'], cfg['m'])
    pytorch_l2_loss = nn.MSELoss()

    if checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_l2_loss.load_state_dict(checkpoint['loss_fn_state_dict'])

    if mode == 'train':
        pytorch_dataset = Noisier2NoiseDataset('../data/' + cfg['dataset_id'] + '/train', 
                                               [cfg['crop_width'], cfg['crop_height']], 
                                               cfg['noise_type'], cfg['noise_std'], mode)
        pytorch_data_loader = data_utils.DataLoader(pytorch_dataset, batch_size=cfg['batch_size'], shuffle=True)
        # TODO(ff-k): Change learning rate based on batch index
        pytorch_optimizer = optim.Adam(pytorch_model.parameters(), lr=cfg['adam_learning_rate_first_phase'], 
                                       betas=(cfg['adam_beta_1'], cfg['adam_beta_2']), eps=cfg['adam_eps'])
        if checkpoint:
            pytorch_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        pytorch_model.train()

        checkpoint_save_interval = 1
        epoch = cfg['epoch']
        while globals()['interruped_by_user'] == False:
            # TODO(ff-k): Noisier2Noise does not use a validation set, but we may 
            #             consider adding that

            time_epoch_begin = perf_counter()
            epoch_loss = 0
            for tensor_noisier, tensor_noisy in pytorch_data_loader:
                prediction = pytorch_model(tensor_noisier)
                loss = pytorch_l2_loss(prediction, tensor_noisy)
                epoch_loss = epoch_loss + loss.item()
                
                pytorch_optimizer.zero_grad()
                loss.backward()
                pytorch_optimizer.step()
            
            avg_epoch_loss = epoch_loss / len(pytorch_data_loader)
            recent_loss = loss.item()
            time_epoch_end = perf_counter()
            print('average loss: %f, recent_loss: %f, elapsed: %f' % (avg_epoch_loss, recent_loss, time_epoch_end-time_epoch_begin));
            if epoch % checkpoint_save_interval == 0 or globals()['interruped_by_user'] == True:
                cfg['epoch'] = epoch + 1
                torch.save({
                    'avg_epoch_loss': avg_epoch_loss,
                    'recent_loss': recent_loss,
                    'run_config': cfg,
                    'model_state_dict': pytorch_model.state_dict(),
                    'optimizer_state_dict': pytorch_optimizer.state_dict(),
                    'loss_fn_state_dict': pytorch_l2_loss.state_dict(),
                    }, checkpoints_path + 'checkpoint_%05d.pt' % (epoch))

            epoch = epoch + 1
    elif mode == 'test':
        pytorch_model.eval()

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        denoised_path = '../denoised/'

        root_path = '../data/' + cfg['dataset_id'] + '/test/'
        img_ext = ['jpg','jpeg', 'bmp', 'png', 'gif']
        img_paths = [fp for fp in os.listdir(root_path) if any(fp.endswith(ext) for ext in img_ext)]

        max_loss = 0
        max_psnr = 0
        max_ssim = 0
        min_loss = float("inf")
        min_psnr = float("inf")
        min_ssim = float("inf")
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        for img_i in range(len(img_paths)):
            img_path = img_paths[img_i]
            img =  im.open(root_path + img_path).convert('RGB')

            # NOTE(ff-k): We have to use some padding to avoid unmatched tensor dimensions.
            #             Basically, we half the size of our input images (via max pool layers)
            #             5 times. Therefore, we have to make sure that image dimensions are 
            #             divisible by 2^5=32.
            pad_w = int((img.size[0]+31)/32)*32-img.size[0]
            pad_h = int((img.size[1]+31)/32)*32-img.size[1]
            if pad_w > 0 or pad_h > 0:
                tmp = im.new(img.mode, (img.size[0]+pad_w, img.size[1]+pad_h))
                tmp.paste(img)
                img = tmp 

            img_noisy = NoiseUtils.apply_noise(img, cfg['noise_type'], cfg['noise_std'])
            tensor_noisy = xform.to_tensor(img_noisy).to(device=device, dtype=torch.float32)
            tensor_noisy = tensor_noisy.unsqueeze(0)

            img_noisier = NoiseUtils.apply_noise(img_noisy, cfg['noise_type'], cfg['noise_std'])
            tensor_noisier = xform.to_tensor(img_noisier).to(device=device, dtype=torch.float32)
            tensor_noisier = tensor_noisier.unsqueeze(0)
            
            prediction = pytorch_model(tensor_noisier)
            prediction_sn = pytorch_model(tensor_noisy)

            if pad_w > 0:
                prediction[:, :, :, -pad_w:] = 0
                prediction_sn[:, :, :, -pad_w:] = 0
                tensor_noisier[:, :, :, -pad_w:] = 0
                tensor_noisy[:, :, :, -pad_w:] = 0
            if pad_h > 0:
                prediction[:, :, -pad_h:, :] = 0
                prediction_sn[:, :, -pad_h:, :] = 0
                tensor_noisier[:, :, -pad_h:, :] = 0
                tensor_noisy[:, :, -pad_h:, :] = 0
            
            loss = pytorch_l2_loss(prediction, tensor_noisy).item()
            if min_loss > loss:
                min_loss = loss
            if max_loss < loss:
                max_loss = loss
            total_loss = total_loss + loss
            
            tensor_residual = torch.sub(tensor_noisier[0], prediction[0])
            tensor_denoised = torch.sub(prediction[0], tensor_residual)
            tensor_denoised = torch.clamp(tensor_denoised, 0.0, 1.0)

            img_clean_01 = np.array(img, dtype=np.float32)/255
            img_predt_01 = np.transpose(tensor_denoised.detach().cpu().numpy(), (1, 2, 0))
            psnr = sk_psnr(img_clean_01, img_predt_01)
            if min_psnr > psnr:
                min_psnr = psnr
            if max_psnr < psnr:
                max_psnr = psnr
            total_psnr = total_psnr + psnr

            ssim = sk_ssim(img_clean_01, img_predt_01, multichannel=True)
            if min_ssim > ssim:
                min_ssim = ssim
            if max_ssim < ssim:
                max_ssim = ssim
            total_ssim = total_ssim + ssim

            if np.random.random() < save_prob:
                save_id = np.random.randint(2147483647)

                tensor_residual_min = torch.min(tensor_residual)
                tensor_residual_max = torch.max(tensor_residual)
                tensor_residual_rng = tensor_residual_max - tensor_residual_min
                tensor_residual_avg = (tensor_residual_max + tensor_residual_min)/2.0
                
                tensor_residual = torch.sub(tensor_residual, tensor_residual_avg)
                tensor_residual = torch.mul(tensor_residual, 1.0/tensor_residual_rng)
                tensor_residual = torch.add(tensor_residual, 0.5)
                tensor_residual = torch.clamp(tensor_residual, 0.0, 1.0)

                img_noisy = xform.to_pil_image(tensor_noisy[0])
                img_noisier = xform.to_pil_image(tensor_noisier[0])
                img_residual = xform.to_pil_image(tensor_residual)
                img_denoised = xform.to_pil_image(tensor_denoised)

                img_noisy.save(os.path.join(denoised_path, '%010d_noisy.png' % (save_id)))
                img_noisier.save(os.path.join(denoised_path, '%010d_noisier.png' % (save_id)))
                img_residual.save(os.path.join(denoised_path, '%010d_residual.png' % (save_id)))
                img_denoised.save(os.path.join(denoised_path, '%010d_denoised.png' % (save_id)))
            
        avg_loss = total_loss / len(img_paths)
        avg_psnr = total_psnr / len(img_paths)
        avg_ssim = total_ssim / len(img_paths)

        print('average loss: %f, min loss: %f, max loss: %f' % (avg_loss, min_loss, max_loss));
        print('average psnr: %f, min psnr: %f, max psnr: %f' % (avg_psnr, min_psnr, max_psnr));
        print('average ssim: %f, min ssim: %f, max ssim: %f' % (avg_ssim, min_ssim, max_ssim));

    else:
        raise NotImplementedError('Unknown running mode: \'%s\'' % mode)

if __name__ == "__main__":

    mode = 'test'
    force_dataset = ''
    save_prob = 0.9011

    argc = len(sys.argv)
    if argc > 3:
        mode = sys.argv[1]
        force_dataset = sys.argv[2]
        save_prob = sys.argv[2]
    elif argc > 2:
        mode = sys.argv[1]
        force_dataset = sys.argv[2]
    elif argc > 1:
        mode = sys.argv[1]
    
    if mode == 'train':
        globals()['interruped_by_user'] = False
        signal.signal(signal.SIGINT, interrupt_handler)
    
    # NOTE(ff-k): set random_seed to 0 or some other constant value to repeat results
    random_seed = int(time.time()*1000)%(2**32-1) 
    np.random.seed(random_seed)
    
    main(mode, force_dataset, save_prob)
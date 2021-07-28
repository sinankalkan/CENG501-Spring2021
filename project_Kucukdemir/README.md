# Noisier2Noise: Learning to Denoise from Unpaired Noisy Data

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

This is an unofficial implementation of [Noisier2Noise by Moran et al. 2020](https://arxiv.org/abs/1910.11908).

## 1.1. Paper summary

In some fields, for example in astrophotography, radiology (MRI) and computer graphics (Monte Carlo renderings), it is difficult to obtain clean images whereas obtaining a noisy one is much easier. Image denoising is the process where we remove noise from such images. After the rise of deep learning techniques in 2012, people began to use CNNs for this purpose and proposed various methods (see [Tian et al. 2020](https://arxiv.org/abs/1912.13171)). 

In CVPR 2020, Moran et al. proposed Noisier2Noise, which is mainly based on a previous method called [Noise2Noise (Lehtinen et al. 2018)](https://arxiv.org/abs/1803.04189). The main contribution of Noisier2Noise is that it does not require clean image targets to train the network, whereas most deep learning based image denoising techniques (including Noise2Noise) do need them at training stage.

# 2. The method and my interpretation

## 2.1. The original method

<p align="center">
  <img src="readme_fig/method.png">
  Figure 1. The brief overview of the method. Taken from [Moran et al. 2020](https://arxiv.org/abs/1910.11908)
</p>

In summary, the training step works as follow:
* We first apply some noise to the clean image. Note that our network will not see the clean image directly.
* Then, we apply noise one more time to get doubly-noisy realization of the clean image.
* Then, we feed the network with doubly-noisy realization and try to predict singly-noisy realization.

During inference, we give the doubly-noisy realization to our network and expect it to output a singly-noisy one. Then, we compute the residual by subtracting singly-noisy realization from doubly-noisy one. In the last step, we subtract the residual from singly-noisy realization and obtain an estimate of the clean target image.

The network architecture used in this method is [U-Net by Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597).

## 2.2. My interpretation 

** TODO mention max epoch and learning rate thing here **

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

**Dependencies**

* Python (3.7.0)
* Numpy (1.19.5)
* PIL (8.3.1)
* PyTorch (1.9.0+cu111) + torchvision (0.10.0+cu111)
* scikit-image (0.18.2)

_(Versions written in parentheses stand for the ones used during the development and testing of this repository)_

I've tried to keep configurations the same as much as possible. However, there is one significant difference. In the original paper, authors have used a batch size of 32. But in this work, it is reduced to 16 due to hardware limitations. 

## 3.2. Running the code

File structure of the project is as follows:

```
checkpoints
│   │   (*.pt)
code
│   │   main.py
data
└─── <dataset_id>
  └─── test
  │   │   image files (*.jpg, *.png etc.)
  └─── train
  │   │   image files (*.jpg, *.png etc.)
denoised
│   │   image files (*.jpg, *.png etc.)
```

**checkpoints**: This folder contains checkpoints (i.e. metadata of your model) that allow you to resume training your network or to use it for testing purposes. The id of checkpoints show the number of epochs. For example, _checkpoint_00010.pt_ contains metadata of a model trained for 11 epochs.

**code**: This folder contains the code and you are supposed to run the project here.

**data**: This folder contains images. You should keep the file structure as shown above. Simply download the dataset, create a folder for it under this directory, create two separate folders (i.e. test and train) and place images under these directories.

**denoised:** This folder contains denoised images (i.e. the output).

**How to Run**

TODO

## 3.3. Results

The code is tested on a Windows 10 (v19042.1110) machine with single NVIDIA GeForce RTX 3070 GPU.

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

* N. Moran, D. Schmidt, Y. Zhong and P. Coady, "Noisier2Noise: Learning to Denoise from Unpaired Noisy Data", CVPR, 2020.
* J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala and T. Aila, "Noise2Noise: Learning Image Restoration without Clean Data", 2018.
* T. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick and P. Dollár, "Microsoft COCO: Common Objects in Context", 2015.
* J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database". IEEE Computer Vision and Pattern Recognition (CVPR), 2009.
* D. Martin, C. Fowlkes, D. Tal and J. Malik, "A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics, Proc. 8th Int'l Conf. Computer Vision, 2001.
* C. Tian, L. Fei, W Zheng, Y. Xu, W. Zuo and C. Lin, "Deep Learning on Image Denoising: An overview", 2020
* O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015.

# Contact

_Please feel free to contact me by email, if you need any further information or have any questions (furkankdem [at] gmail.com)_

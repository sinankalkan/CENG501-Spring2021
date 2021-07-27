# Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

In this repository, we describe and try to implement some of the experiments on the "Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary" paper published on Computer Vision and Pattern Recognition Conference, 2020. This paper deals with the two biggest problems with medical image segmentation: uncertainty of the boundary of the structure in medical images and the uncertainty of the partitioned region without special domain information. The paper offers a solution to these problems by a boundary key point selection algorithm and shape boundary-aware evaluator. We aim to reproduce the experiments implemented in the article and obtain similar results.

## 1.1. Paper summary

For medical image processing applications, segmenting the anatomical structures is correctly is an important task. In the existing literature, Fully Convolution Networks and U-Net used to semantic segmentation tasks. However, it has been observed that these deep learning-based medical image segmentation methods cannot successfully segment difficulties in the medical image such as unclear structure boundaries and heterogeneous textures during segmentation. Figure 1 shows cases where U-Net fails to segment target regions in the ultrasound image.
<p align="center">
  
![Adsız](https://user-images.githubusercontent.com/82761420/127210354-32dec827-0fed-4fe2-96f3-b3befded14bc.png)
</p>
<p align="center">
  
Figure 1: U-Net segmentation results which are failed in preserving structure boundary
  </p>
In the article, two problems that arise in segmentation in medical images are tried to be overcome. The first of these problems is that medical images contain unclear borders due to poor quality and heterogeneous textures. The other problem is that it is difficult to automatically predict the target area without expert knowledge. To overcome these problems, tha paper proposed one algorithm, one network structure and one discriminator structure : 

One of the algorithms is boundary key point selection algorithm, which aims at finding key points on the boundary of ground truth segmentation map. 

Boundary Preserving Block (BPB) is a unit on CNN that enhances the boundary information of input. 

Shape Boundary- aware Evaluator (SBE) is a discriminator which is only used during training. It discriminates and outputs an evaluation score between the predicted segmentation map concatenated with boundary key point map & ground truth segmentation map concatenated with boundary key point map. 

The biggest contribution of this paper is the proposed method can be generalized to the different segmentation models. 

# 2. The method and my interpretation

## 2.1. The original method

Explain the original method.

## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

Explain your code & directory structure and how other people can run it.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Murat Kara muratka058@gmail.com
İlayda Altun altunilayda8@gmail.com

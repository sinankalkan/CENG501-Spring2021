# Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

In this repository, we describe and try to implement some of the experiments on the "Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary" paper published on Computer Vision and Pattern Recognition Conference, 2020. This paper deals with the two biggest problems with medical image segmentation: uncertainty of the boundary of the structure in medical images and the uncertainty of the partitioned region without special domain information. The paper offers a solution to these problems by a generic solution. This solution makes the efforts of detecting the boundaries fully automatized. The idea has given two novel methods (Boundary Preserving Block Maps and Structured Boundary Evaluator) to the community. We aim to reproduce the experiments implemented in the article and obtain similar results. 

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

The major contribution of this paper on literature is the proposed method can be generalized to the different segmentation models. 

# 2. The method and my interpretation

## 2.1. The original method
This section describes the original method used in paper. Also, dataset and some implementation details are given. 

### 2.1.1 BPB & SBE
In the original method, the boundary preserving segmentation structure consists Boundary Preserving Block (BPB) and Shape Boundary-aware Evaluator(SBE). The BPB and SBE use the boundary key points which are selected from boundary key point selection algorithm.
#### 2.1.1.1 Boundary Key Point Selection Algorithm
The algorithm select the boundary key points that best fit the ground truth segmentation map. The algorithm works as described below :   
1- Obtain the boundary of the target object using edge detection  
2- On the boundary, randomly select n points  
3- These n points connected to form a polygon.  
4- Repeat for T times and select the key point sets which have the highest intersection of union.  
#### 2.1.1.2 Boundary Preserving Block (BPB)
This unit can be embedded into any network and it enhances the boundary information of input. Boundary Preserving Block predicts the segmentation map.  This unit consist boundary Point Map Generator, that produce a key point prediction map. The generator consist boundary key point selection module which generates ground truth boundary key point maps. It also consist dilated convolution, so the generator can effectively encode & decode the features with a various range of receptive fields. Generator optimized by cross entropy loss between the estimated boundary key point map and ground truth boundary key point map. 

#### 2.1.1.3 Shape Boundary-aware Evaluator (SBE)
SBE is an evaluator which gives feedback to the network by using the bundary key point map. Basically, its inputs are boundary key point map and predicted or ground truth segmentation image, and it evaluates whether the segmentation results are consistent with the boundary key point map or not. 

### 2.1.2 Dataset
The paper used the PH2 and ISBI 2016 datasets. These datasets are publically available for skin lesion segmentation.  ISBI 2016 dataset consists 900 skin lesion images, PH2 includes 200 dermoscopc images. They used ISBI 2016 dataset for training and PH2 for testing. They also used TVUS dataset which is private and collected for experiments. 

### 2.1.3 Implementation Details

-The model optimized by using ADAM optimizer.  
-The learning rate was 0.0001 both in segmentation network and SBE network.  
-They used randomly initialized weights with 8 input batches.  
-Forboundary key point maps , they selected 6 points and run the algorithm 40000 times. 
-They integrated the proposed method in several segmentation networks : U-Net , FCN, Dilated-Net. 


## 2.2. Our interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.
First , we wanted to have the same segmentation scores on different networks. We tried to implement the FCN segmentation network with using ISBI 2016 dataset. Unfortunately, we couldn't get the segmentation scores yet. The FCN networks which are using ISBI 2016 dataset included on the FCN codes file. 

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

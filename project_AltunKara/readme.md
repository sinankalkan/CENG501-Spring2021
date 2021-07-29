# Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

In this repository, we describe and try to implement some of the experiments on the "Structure Boundary Preserving Segmentation for Medical Image with Ambiguous Boundary" paper [1] published on Computer Vision and Pattern Recognition Conference, 2020. This paper deals with the two biggest problems with medical image segmentation: uncertainty of the boundary of the structure in medical images and the uncertainty of the partitioned region without special domain information. The paper offers a solution to these problems by a generic solution. This solution makes the efforts of detecting the boundaries fully automatized. The idea has given two novel methods (Boundary Preserving Block Maps and Structured Boundary Evaluator) to the community. We aim to reproduce the experiments implemented in the article and obtain similar results. 

## 1.1. Paper summary

For medical image processing applications, segmenting the anatomical structures is correctly is an important task. In the existing literature, Fully Convolution Networks and U-Net are used for semantic segmentation tasks. However, it has been observed that these deep learning-based medical image segmentation methods cannot successfully segment difficulties in the medical image such as unclear structure boundaries and heterogeneous textures during segmentation. Figure 1 shows cases where U-Net fails to segment target regions in the ultrasound image.
<p align="center">
  
![Adsız](https://user-images.githubusercontent.com/82761420/127210354-32dec827-0fed-4fe2-96f3-b3befded14bc.png)
</p>
<p align="center">
  
Figure 1: U-Net segmentation results which are failed in preserving structure boundary
  </p>
In the article, two problems that arise in segmentation in medical images are tried to be overcome. The first of these problems is that medical images contain unclear borders due to poor quality and heterogeneous textures. The other problem is that it is difficult to automatically predict the target area without expert knowledge. To overcome these problems, the paper proposed one algorithm, one network structure, and one discriminator structure :

One of the algorithms is the boundary key point selection algorithm, which aims at finding key points on the boundary of the ground truth segmentation map. 

Boundary Preserving Block (BPB) is a unit on CNN that enhances the boundary information of input. 

Shape Boundary- aware Evaluator (SBE) is a discriminator which is only used during training. It discriminates and outputs an evaluation score between the predicted segmentation map concatenated with the boundary key point map & the ground truth segmentation map concatenated with the boundary key point map. 

The major contribution of this literature paper is the proposed method can be generalized to the different segmentation models. 

# 2. The method and my interpretation

## 2.1. The original method
The original method uses a Google-Net-like structure in the train segmentation part and a GAN-like structure in the SBE part.  These structures are an extension to already existing image segmentation algorithms like U-net. In the train segmentation part, the existing method is extended by Boundary Preserving Blocks that give some output and have a loss function like in the Google-Net. In the SBE part, there is a discriminator network that discriminates the ground truth segmentation map image as real and predicted segmentation map image as fake by comparing them with boundary preserving map. In this way, network forces taking boundary information into account better in the predicted segmentation map. The SBE part is only used in the training. 

### 2.1.1 Building Blocks
In the original method, the boundary preserving segmentation method has some building blocks. The structure consists of Boundary Key Point Selection Algorithm, Boundary Preserving Block (BPB), and Shape Boundary-aware Evaluator(SBE). 

#### 2.1.1.1 Boundary Key Point Selection Algorithm
The algorithm selects the boundary key points that best fit the ground truth segmentation map. In this way, we expect these points to describe the boundaries well. The algorithm works as described below :   
1- Obtain the boundary of the target object using edge detection  
2- On the boundary, randomly select n points.
3- These n points are connected to form a polygon.  
4- Repeat for T times and select the key point sets which have the highest intersection of union with the ground truth segmentation map. 
5- To increase the chance of coinciding edge points, we take a disk with points as center and R as radius instead of only points. This is our boundary key point selection.

#### 2.1.1.2 Boundary Preserving Block (BPB)

This block is a structure that is used in the training segmentation part. It can be embedded into any network and it enhances the boundary information of input. This unit is added after some layers. It takes the features of the previous layer and applies different levels of dilation to them. Then, the outputs are concatenated, 1X1 dilation and sigmoid are applied to that concatenated outputs. The benefit of dilated convolution is that the network can effectively encode & decode the features with a various range of receptive fields. Then, the loss is applied with boundary key points(actually disks) selected by the boundary key point selection algorithm. The output of this learned parameter is multiplied with features of the previous layer and the features are added again (vi = (fi * Mi) + Mi). This value is passed to the next layer.  

#### 2.1.1.3 Shape Boundary-aware Evaluator (SBE)

SBE is an evaluator that evaluates the difference boundary key point map with ground-truth segmentation map and predicted segmentation map. The difference with ground-truth segmentation can be thought of as real input as in GAN architecture. So, the difference is expected a high score. On the other hand, the difference with a lowly predicted segmentation map gets a low score since the predicted segmentation will not be consistent with the boundary of the image in this situation. So, the network gives feedback to the network by using the boundary key point map to update the predicted segmentation map. The two networks are trained iteratively.   

### 2.1.2 Dataset
The paper used the PH2 and ISBI 2016 datasets. These datasets are publically available for skin lesion segmentation.  ISBI 2016 dataset consists of 900 skin lesion images, PH2 includes 200 dermoscopic images. They used ISBI 2016 dataset for training and PH2 for testing. They also used the TVUS dataset which is private and collected for experiments.  
ISBI2016 : [ISBI2016](https://challenge.isic-archive.com/data)  
PH2 : [PH2](https://www.fc.up.pt/addi/ph2%20database.html)

### 2.1.3 Implementation Details

- The model is optimized by using an ADAM optimizer.  
- The learning rate was 0.0001 both in the segmentation network and SBE network.  
- They used randomly initialized weights with 8 input batches.  
- For boundary key point maps, they selected 6 points and run the algorithm 40000 times. 
- They integrated the proposed method in several segmentation networks: U-Net, FCN, Dilated-Net. 
- For every iteration, they train the segmentation network 8 times and the SBE network 3 times to train two networks in an adversarial manner.


## 2.2. Our interpretation 

First, we wanted to have the same segmentation scores on different networks. We tried to implement the FCN segmentation network using ISBI 2016 dataset. Unfortunately, we couldn't get the segmentation scores yet. 
The FCN networks which are using ISBI 2016 dataset are included in the FCN codes file.  
- In the FCN_Implementation of ISIC_2016 file, which is a modified version of an FCN network [2]. In this network, there were several classes but in our case, there is only 2 classes. We got the results but the problem is the predicted images are fully black. We believe that is because of two classes exist. So predicted images represent only one class.  We still try to fix this issue.    
- In the FCN_ISIC2016.ipynb file, even though the training is working, it takes so much time, we can't see the results.  


On the other hand, we successfully implement the U-Net segmentation model to the ISBI 2016 dataset. We used mostly a medium page that explains the U-Net structure [3] and modified it for the datasets. Our modified U-Net code is available as .py and .ipnyb files but there must be a configuration of data sets. We could not upload folders. So, the dataset paths must be changed inside the code. It is the -wrong- implementation of the first training part: U-Net  + BPB 

Afterward, blocks of our algorithm are added. Firstly, it was decided to start with Boundary Preserving Block. For edge detection, Canny Edge Detection is used from the OpenCV library. For constructing the polygon, the points are needed to be sorted in the counter clock-wise order. In the end, we got the polygons but we made a mistake that is realized at the deadline. We took the polygon instead of the points as the output of the algorithm.

Because we did not realize it, we continued. That polygon was used in the boundary preserving block implementation instead of the boundary key points map. This way, unet_plus_bpbm_net,i.e first part of the network, was implemented. Note that, the boundary key point selection algorithm creates a fixed-sized output. For using these boundary key points for these layers, we needed to downsample them 6 times. bpbm_creator.py is for creating the boundary key points polygon, which should have been pointing instead, and its downsamples. Also, the SBE network was created by looking at the supplementary part of the paper. However, there was an inconsistency while creating the full network. The loss functions were not consistent. After trying a while, we finished our hope and realized our mistake, after a few more times. 

In actual implementation, the two parts of the network are not connected, To connect them logically, there is a boundary-aware loss in the first network in addition to SBE's loss in the second part. 

We tried lots of things but that did not work as we said before, we put as it is. The main file is coding22.py.

# 3. Experiments and results

## 3.1. Experimental setup

In the implementation details, most of the experimental setup info exists. 
While training we would have changed the filter sizes. Dataset settings were public ones(ISBI2016 as training and validation, PH2 as testing) in our earlier experiments.

## 3.2. Running the code

coding22.py would have run and we would have seen the results. But it is broken now. But, it is possible to run the unet_denemeler by changing dataset directories.

## 3.3. Results

We did not see the results. However, We implemented the first part of the network(unet_denemeler). When We trained and test it, We saw that there was noise inside the predicted segmentation images inside the lesions and overflows in the segmentation. The first problem may have arisen from my mistake of using a polygon. So, We added noise to the polygon part of the map via boundary preserving block map training. The second problem may have come from a lack of SBE that optimizes to catch the boundaries.

# 4. Conclusion

In conclusion, even though we couldn't conclude the code and reached the results, we tried to implement a brand new method on existing segmentation network structures. We have seen complex and beneficial thinking of networks.

# 5. References

Provide your references here.
[1] https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Structure_Boundary_Preserving_Segmentation_for_Medical_Image_With_Ambiguous_Boundary_CVPR_2020_paper.pdf
[2] https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html
[3] https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5

# Contact

Murat Kara muratka058@gmail.com
İlayda Altun altunilayda8@gmail.com

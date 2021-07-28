# Object-Guided Instance Segmentation for Biological Images[1]

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper we tried to implement is titled Object-Guided Instance Segmentation for Biological Images and published in Proceedings of the AAAI Conference on Artificial Intelligence 34. The authors suggest a box-based instance segmentation method for biological images. 

## 1.1. Paper summary
Box-based instance segmentation methods use Region of Interest (RoI) patches generated from bounding boxes. These bounding boxes are obtained by examining all the pixels in an image. After obtaining the bounding boxes RoI patches are generated. The Keypoint Graph method[2] uses a separate segmentation branch to operate on cropped RoI patches. In the segmentation branch five keypoints are detected for generating a bounding box, which creates problems when objects are overlapping. In this paper authors suggest using a keypoint based method where only the center point is used to generate each bounding box for overcoming the problem of overlapping keypoints at the detection branch. 

# 2. The method and my interpretation
Proposed method is given below:
<p align="center">
	<img src="figures/method.png", width="800">
</p>

## 2.1. The original method

There are two branches on network architecture. One is to detect object and create the bounding boxes, other is to segmentation.

### 2.1.1 Detection Branch
In first branch a U shaped architecture used with skip layers (See [4] ). Skip layers made the layer size double at each combine operation. U shaped network and Skip layer schema is given below:
<p align="center">
	<img src="figures/detBranch.png", height="300">
	<img src="figures/skip.png", height="300">
</p>

The output of the U shaped network is used for three operation: Heatmap generation, Bounding Box Width Height, and Bounding Box Offset generation. 
<p align="center">
	<img src="figures/outputDec.png", width="400">
</p>
2.1.1.1 Heatmap:

Each object has only one ground truth center location. The authors suggest using a Gaussian circle around each center point following the work of [5] and using the variant focal loss.

<p align="center">
	<img src="figures/lossHM.png", width="600">
</p>

Where _i_ indexes _i_ th location in the predicted heatmap _N_ is the total number of center points, _y_ is the ground truth. We use α = 2, β = 4 in this paper.The predicted center heatmaps are refined through a non-maximum-suppression (NMS) operation. The operation employs a 3×3 max-pooling layer on the center heatmaps. The center points are gathered according to their local maximum probability.

2.1.1.2 Offset Map: 

The center points are predicted from downsized heatmaps. When mapping these points from downsized images to their original locations an offset map is needed. The authors used L1 loss for regressing the offset center points.

2.1.1.3 Width-Height Map: 

Width and height of bounding boxes are obtained from heatmaps. The authors used L1 loss to regress width and height of bounding boxes.


## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

* We did not know if there is a filter before Heatmap, Width-Height, and Offset Heads. In [newer paper](https://arxiv.org/abs/2106.07159)[3] there is filters before each head. But in the code the heads are different. In the code Width-Height and Offset modules filters are swapped so there are 7x7 layers before WH  and 3x3-1x1 layers before offset head. 
<p align="center">
	<img src="figures/DecHead.png", width="400">
</p>

* Non-linearities are never mentioned in paper. However, in code after every convolutional layer there is a ReLU non-linearity Layer.

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

[1] [Jingru Yi, Hui Tang, Pengxiang Wu, Bo Liu, Daniel J. Hoeppner, Dimitris N. Metaxas, Lianyi Han, Wei Fan. Object-Guided Instance Segmentation for Biological Images, 2019.](https://arxiv.org/abs/1911.09199) 

[2] [Jingru Yi, Pengxiang Wu, Qiaoying Huang, Hui Qu, Bo Liu, Daniel J. Hoeppner, Dimitris N. Metaxas. Multi-scale Cell Instance Segmentation with Keypoint Graph based Bounding Boxes, 2019.](https://arxiv.org/abs/1907.09140) 

[3] [Jingru Yi, Pengxiang Wu, Hui Tang, Bo Liu, Qiaoying Huang, Hui Qu, Lianyi Han, Wei Fan, Daniel J. Hoeppner, Dimitris N. Metaxas.Object-Guided Instance Segmentation With Auxiliary Feature Refinement for Biological Images. 2021](https://arxiv.org/abs/2106.07159)

[4] [Olaf Ronneberger, Philipp Fischer, Thomas Brox.U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015](https://arxiv.org/abs/1505.04597)

[5] [Hei Law, Jia Deng.CornerNet: Detecting Objects as Paired Keypoints.2018](https://arxiv.org/abs/1808.01244)

# Disclaimer

We did not write the code ourselves. The code belongs to [Jingru Yi](https://github.com/yijingru/ObjGuided-Instance-Segmentation). Code of the paper is not shared. [An improved version of the paper](https://arxiv.org/abs/2106.07159)[3] was published in IEEE Transactions on Medical Imaging. We used that paper's code and deleted the new parts of the paper.

# Contact

Yusuf Can Aydemir : can.aydemir@metu.edu.tr

Yunus Bilge Kurt : yunusbilgekurt@gmail.com

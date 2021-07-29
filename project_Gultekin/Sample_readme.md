# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

In this Project, a CVPR 2020 published paper named Density Map Guided Object Detection in Aerial Images is considered to make implementation. Autors introduce a new cropping strategy in the paper using CNN. 
My goal is that reproduce Density Map using CNN to use them for cropping strategy and then detect object in aerial images 


## 1.1. Paper summary

Object detection in high-resolution aerial images is a challenging task because of 1) the large variation in object size, and 2) non-uniform distribution of objects [1]. It is hard to detect small objects in aerial images. Cropping images is the method that applied before the object detection algorithm such as uniform cropping [2]. For most of the cases, the method help improve the detection accuracy of small objects. However, these simple strategies  are not able to leverage the semantic information  for cropping, thus resulting in a majority of crops with only background and large objects may be cut into two or more different crops by these strategies.[1]

Density Map Network -DMNet provide semantic cropping. It utilizes object density map to indicate the presence of objects as well as the object density within a region. The distribution of objects enables our cropping module to generate better image crops for further object detection as shown fig.1. 

![univsdensity](https://user-images.githubusercontent.com/48828422/127484682-3f2f07e8-4db2-4e77-9686-0b05914dd644.PNG)

Figure 1: Uniform cropping vs. Density cropping [1]

Two popular aerial images datasets VisionDrone [3] and UAVDT [4] was used in the paper.


# 2. The method and my interpretation

## 2.1. The original method

![framework](https://user-images.githubusercontent.com/48828422/127486390-72d82226-2ed0-4b56-a700-23ead73165b0.PNG)

Figure 2: Overview for the DMNet framework. [1]

DMNet Generation : Density map is of great significance in the context of crowd counting [5] proposes the Multi-column CNN (MCNN) to learn density map for crowd counting tast. Due to the variation of head size per image, single column with fixed receptive field may not capture enough features. Therefore three columns are introduced to enhance feature extraction [1]. Author of the paper adopt MCNN [5] in their approach to generate object density map for image cropping. 

The Loss function for training density map generation network is based on the pixel-wise mean absolute error, which is given as below: 

![loss](https://user-images.githubusercontent.com/48828422/127487939-3e493e57-5037-49e8-97a7-97795fc72ffc.PNG)
[1]


Θ is the parameters of density map generation module. N is the total number of images in the training set. Xi is the input image and Di is the ground truth density map for image Xi. D(Xi;Θ) stands for the generated density map by the density generation network. 

![MCNN](https://user-images.githubusercontent.com/48828422/127489814-c4533ae5-c75c-463c-b4bb-20b759a4acff.PNG)

Figure 3: MCNN framework. [5]

Ground truth object density map : To generate the ground truth object density maps for aerial images in the training stage, they follow geometry-adaptive and geometry-fixed kernel [5].

![ground truth](https://user-images.githubusercontent.com/48828422/127490872-dd2aeedf-8392-47f1-b9a2-ef96f7877778.PNG)

Figure 4: Original images and corresponding crowd density maps obtained by convolving geometry-adaptive Gaussian kernels [5].



## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

Their implementation is based on the MMDetection toolbox [6]. The MCNN [5] is selected as the baseline network for density map generation. For object detector, They use Faster R-CNN with Feature Pyramid Network (FPN). Unless specified, they use the default configurations for all the experiments. They use ImageNet [7] pre-trained weights to train the detector. The density threshold is set to 0.08 in both training and testing phases for VisionDrone dataset and 0.03 for UAVDT dataset. The minimal threshold for filtering bounding boxes is set to 70 × 70, which follows the similar setting in [8]. 

The density map generation module is trained for 80 epochs using the SGD optimizer. The initial learning rate is 10−6. The momentum is 0.95 and the weight decay is 0.0005. They only use one GPU to train the density map generation network and no data argumentation is used.

For the object detector, they set the input size to 600 × 1,000 on both datasets. They follow the similar setup in [8] to train and test on the datasets. The detector is trained for 42 epochs on 2 GPUs, each with a batch size of 2. The initial learning rate is 0.005. They decay the learning rate by the factor of 10 at 25 and 35 epochs. The threshold for nonmax suppression in fusion detection is 0.7. The maximum allowed number for bounding boxes after fusion detection is 500. Unless specified, they use MCNN to generate density map and Faster R-CNN with FPN to detect objects for all the experiments. [1]

## 3.2. Running the code

Explain your code & directory structure and how other people can run it.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

[1]  Changlin Li, Taojiannan Yang, Sijie Zhu, Chen Chen, Shanyue Guan, Density Map Guided Object Detection in Aerial Images, CVPR 2020 Workshop

[2]  F. Ozge Unel, Burak O. Ozkalayci, and Cevahir Cigla. The power of tiling for small object detection. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

[3]  Pengfei Zhu, Longyin Wen, Xiao Bian, Haibin Ling, and Qinghua Hu. Vision Meets Drones: A Challenge. Apr 2018

[4]  Dawei Du, Yuankai Qi, Hongyang Yu, Yifan Yang, Kaiwen Duan, Guorong Li, Weigang Zhang, Qingming Huang, and Qi Tian. The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking. arXiv e-prints, page arXiv:1804.00518, Mar. 2018.

[5]  Y. Zhang, D. Zhou, S. Chen, S. Gao, and Y. Ma. Singleimage crowd counting via multi-column convolutional neural network. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 589–597, June 2016.

[6]  Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.

[7]  Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015.

[8]  Fan Yang, Heng Fan, Peng Chu, Erik Blasch, and Haibin Ling. Clustered object detection in aerial images. In The IEEE International Conference on Computer Vision (ICCV), October 2019.

 
# Contact

Provide your names & email addresses and any other info with which people can contact you.

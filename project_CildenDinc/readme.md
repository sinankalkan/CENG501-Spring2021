# Semi-supervised Semantic Segmentation via Strong-weak Dual-branch Network

This readme file is an outcome of the CENG501 (Spring 2021) project for reproducing a paper which did not have any implementations. See CENG501 (Spring 2021) Project List for a complete list of all paper reproduction projects.

# 1. Introduction

The goal of this study is to reproduce the results of the paper “Semi-supervised Semantic Segmentation via Strong-weak Dual-branch Network”[1], authored by Wenfeng Luo and Meng Yang. The paper was published in the European Conference on Computer Vision in year 2020. We used Google Colab as implementation environment throughout our project. 

## 1.1. Paper summary
### 1.1.1. Objective of the study

Researchers’ main objective is to utilize the weakly supervised (image level labeled) data in image segmentation tasks. In semantic segmentation problem, convolutional neural networks (CNNs) are observed to be very powerful but they need mass amount of annotated data to be precise. However, obtaining strongly annotated data (pixel-level annotated data) is very time consuming and expensive. The study showed that training single branch networks with strong and weak annotations  even decreases the accuracy compared to networks trained with only  strongly annotated data. 

![image](https://user-images.githubusercontent.com/85442860/127391765-e8caea20-eef9-4ade-8d22-bcc05bbd0747.png)

Figure 1: Plot from the paper [1] showing performance of the single branch network with different datasets for 150 epochs

In the figure the performance of 3 different combinations of datasets using the same single branch network is shown. The blue line represents the results achieved via strong 1.4 k data, red line represents the performance gathered by using both 1.4k strong and 9k weak data, and the green line represents the results of 10k weak data only. The best performance is obtained using strongly annotated 1.4k data. However the performance of the task with strong 1.4 k and weak 9k data is substantially lower than the first one. Furthermore, the result obtained with using only weakly annotated 10k data and no strongly annotated data is approximately close to the results of the strong and weak combination. This shows that for a single branch network architecture, weakly annotated data has no use in increasing the accuracy of the semantic image segmentation task and it also downgrades the performance gained with just a small strongly annotated data in a single branch network architecture. 
Authors mentioned that the simple combination of strong and weak annotations with equal treatment may weaken the final performance and so they tried different methods to combine strong and weak data in various ways to increase the accuracy. Authors followed a different perspective: They used a dual branch network and fed two branches simultaneously via using different combinations of strong and weak data. The details of the original method are given in 2.1.
### 1.1.2. Contributions

Researchers mention about 3 main contributions of their paper [1]:
1.	They for the first time showed that segmentation network trained under mixed strong and weak annotations achieves even worse results than using only the strong ones.
2.	They revealed that sample imbalance and supervision inconsistency  between strong and weak data are two key obstacles in improving the performance of semi supervised semantic segmentation.
3.	They proposed a simple unified network architecture to address the inconsistency problem of annotation data in the semi-supervised setting. 

# 2. The method and our interpretation

## 2.1. The original method
Researchers showed the effect of using weakly annotated dataset across the strongly annotated data with different architectures. They illustrated the results in the figure shown below:

![image](https://user-images.githubusercontent.com/85442860/127392075-2ea8ee9f-de9e-4a3f-ac3d-10a1c6f440cd.png)

Figure 2: Single Branch figure from the paper [1]

Fig. 2 (e) shows the single branch network architecture they used. Fig 2.(a) is the test image. Fig 2.(b) shows the effect of strong and weak data together in a single-branch network. Firstly they used a single branch network and they fed it with both strong and weak data. The model cannot distinguish the area (background) between trains from the objects. They also fed the same single branch network with strong annotations and obtained the segmentation result in (c), which is much more successful than (b) showing that utilizing weak annotations besides the strong annotations even makes the accuracy worse. Fig 2. (d) shows the result of using a strong-weak dual-branch network that is much better than (b) and is slightly better than (d). These experiments show that using weakly annotated data brings no improvement with a single branch network but it brings an improvement on the results obtained using only strongly annotated dataset when the data is fed to a dual- branch network. 

Figure 3 presents the dual-branch architecture proposed in the paper. The architecture consists of three main parts. The first part is the pretrained backbone VGG-16, which is followed by a neck module of n layers (n is max. 3). The last module is composed of two identical branches that are trained on strong and weak parts of input data respectively.

![image](https://user-images.githubusercontent.com/85442860/127392145-257e05d8-5c5e-4bc3-9ee3-25fbbf276989.png)

Figure 3: Architecture of the dual-branch network (taken from the paper)[1]

They trained the network on PASCAL VOC dataset [3] and used cross-entropy loss as shown in Figure 4. They generated weak data and proxy ground truth by using the weakly-supervised Deep Seeded Region Growing (DSRG) method [2]. They used random scaling and horizontal flipping as data augmentation and the image batches are cropped to 328x328.
They used AdamOptimizer with an initial learning rate of 5e-6 for the backbone and 1e-4 for the branches. The learning rate decayed by a factor of 10 after 12 epochs. They trained the network for 20 epochs, with a batch size of 16 and weight decay 1e-4. They evaluated the network on both PASCAL VOC and COCO dataset with the standard mean Interaction over Union (mIoU) metric.

![image](https://user-images.githubusercontent.com/85442860/127392215-8bcf81ef-27d6-451d-8914-8c614a8607e6.png)

Figure 4. Loss computation (taken from the paper [1])

## 2.2. Our interpretation 

We started by generating our weak data from DSRG for training. The details of this study are given in Section 3.1.1. To replicate the results of the paper, we implemented both a single branch network and a dual branch network for semantic segmentation task. We trained the single branch network twice. One with only strongly annotated data, the other with both weak and strong annotations. 

Some implementation details were not clearly explained in the original paper. Here we are mentioning of our assumptions about these parts: 
For strongly annotated data, authors used PASCAL VOC 2012 train set (1.4k) and we used this as well. To generate weakly annotated data, authors utilized from Semantic Boundary Detection Dataset [4] (will be called as SBD dataset from now on) and fed it to DSRG to get weakly annotated 9k data. However, in the website of SBD dataset it is mentioned that “The SBD currently contains annotations from 11355 images taken from the PASCAL VOC 2011 dataset.” We could not clearly understand which portion of SBD dataset is used during the weak dataset generation. In the website of SBD, there is an expression saying that “If you are interested in testing on VOC 2012 val, then use this train set, which excludes all val images. This was the set used in our ECCV 2014 paper. This train set contains 5623 images.” Because of this reason we used this partition to generate second part of the weakly annotations. In total we had 7072 weakly annotated data. In the paper they had more weakly annotated data (maximum of 10k in different cases) but since we had limited time and GPU resource we used only 1.4k portion of the weakly annotated data for training the dual branch network.

Another point which is missing in the paper is the training, test and validation partition ratios of the dataset. In our implementation, we did not use any validation set and we adopted standard bisection system. We used the train partition of original PASCAL VOC 2012 dataset as training set and validation partition of the original dataset as the test set. 

In the paper; it is not clear how many layers exist in the single branch network case. We assumed this as 8 (3 for the neck part, 4 for transposed convolutions, and 1 for 1x1 convolution). Also the downsampling and upsampling details are not specified. Input is fed to backbone first and then until neck we assume that it is downsized upto some number. After neck, during the branch parts in both single branch and dual branch networks, we assumed that images are upsampled and at the end the size of the output images (mask) is equal to the size of the inputs. Since they did not provide any details about the upsampling approach they used (like transposed convolution, or bilateral interpolation and number of layers they trained the branches with) we implemented these parts with transposed convolution.

In the paper, VGG16 network is used. Due to our limited GPU resources, (we utilized Google Colab during our implementation) we decided to use ResNet18 network that has much less number of parameters than VGG. 

In the paper, authors used random scaling and horizontal flipping as data augmentation and they cropped image batches to fixed dimension of 328x328. In our implementation, we use random horizontal flipping and we do not use random scaling. We also did not use cropping since we used resizing to 328x328.
We adapted the settings about learning rate and Adam optimizer the same as the Single Branch experiments. For the dual branch, the network size was not appropriate for training in the Colab environment. To decrease the network size, we halved the depth parameters. For obtaining better loss values, we increased the learning rate by a factor of 10 and used an initial learning rate of 1e-3 for the newly-added branches and 5e-5 for the backbone. The learning rate is decayed by a factor of 10 after  12 epochs as in the original paper. We used the same batch size and weight decay parameters.
For the dataset class implementations, we adapted the code from the github page https://github.com/DevikalyanDas/Semantic-Segmentation-of-pascal-voc-dataset-using-Pytorch/blob/main/Pascal_VOC_Segmentation.ipynb [6]. 

### 2.2.1. Single Branch Network

For the backbone, we used pretrained ResNet 18 excluding the last 2 layers. Afterwards, as the paper says, we added a neck part which consists of 3 layers in our case. In the first 2 neck layers we decreased the number of channels to half. In the third neck layer we did not change the number of channels. We decreased the number of channels from the initial value 512 to 128 during the first 2 neck layers. When we tried higher depth values, we came up with a very huge number of parameters after implementing the single branch following the neck layers. Thus, we decided to downsize channels during the neck phase to get less number of parameters in the following phases considering our limited resources with Google Colab. We applied ReLU as nonlinearity at the neck layers. After the neck, we implemented 4 transposed convolutional layers to increase the width and height gradually and at the last transposed convolutional layer we got (328,328) as width and height. Moreover, to be able to map channels to classes (21 PASCAL VOC dataset classes) we placed 1x1 convolutional layer to the end of the single branch. We applied batch normalization at the initial layers of the neck and single branch modules.  We used Xavier initialization for convolutional layers.

### 2.2.2. Dual Branch Network

As in the Single Branch Network, we used ResNet 18 excluding the last 2 layers for the backbone. We added a neck part which consists of 3 layers in our case. Due to Colab limitations, we decreased the number of channels from 512 to 64 in the neck layers. The branches have identical architecture. As in the single branch network, we have 4 transposed convolution and 1 1x1 convolution layers. We applied ReLU non-linearity and batch normalization similarly as in the Single Branch Network. We used Xavier initialization for convolutional layers.    

# 3. Experiments and results

## 3.1. Experimental setup

### 3.1.1. About datasets

As in the paper, we used PASCAL VOC 2012 dataset as the main dataset for strongly annotated data. For weakly annotated data, we downloaded Semantic Boundary Detection (SBD) dataset and generated weakly annotated data using DSRG method as well.  We carried our evaluations on the validation split of the PASCAL VOC 2012 dataset.

PASCAL VOC 2012 dataset includes images for four image-oriented tasks: Object recognition (main), person layout, segmentation, and action detection. For segmentation tasks, there are two alternatives. One of them is object segmentation where different objects are segmented with different coloured masks. The other alternative is class segmentation task where objects belonging to the same class are segmented via the same colored masks. In our case, the researchers of the paper used the class segmentation data partition so we used these images and related annotations as well. 

We used the PASCAL VOC 2012 train dataset as the strongly annotated dataset. We also followed the technique in the paper to get a weakly annotated dataset. We have found a Pytorch implementation for DSRG on github [5]. However, we did not use these codes as it is since we could not run them for our case. We forked this github repo and on our repo and we changed these codes to be able to run them on the Colab environment. Our edited version of DSRG codes is available on our github repo [7]. 

During the construction of weakly annotated datasets since we used DSRG as the generator, we had to train it with PASCAL VOC dataset. In the github repository that we utilized during generation of weakly annotated data, it is recommended to train network for 8000 epochs. However, due to colab restrictions we could not manage to train so long and we trained the network for 1500 epochs as a maximum epoch number at the end. Due to this reason our weakly annotated data is really weak, indeed it is much more weaker than the ones which were used in the paper. 

We used Semantic Boundaries Dataset’s (SBD Dataset) train partition (5623 images) to generate additional 5.6 k weakly annotated data. In total we had 7072 weakly annotated images. We used first 1500 of them as the additional weakly annotated data to train our Single Branch Network. 

### 3.1.2. Experiments with the Single Branch Network

In Single Branch Network, our first experiment was training the network with 1.4 strong data (train partition of PASCAL VOC 2012 dataset). After that, we tested predictions and evaluated our results. In the paper, authors came up with 68.9 mIoU as the performance. We came up with 22.38. Our loss curve was as shown below:

![image](https://user-images.githubusercontent.com/85442860/127392870-621cc72d-aac0-42dd-b7d9-b7ecaa75c18e.png)

Figure 5. Loss curve for Single Branch Network trained with strong annotations

The second experiment we performed with this network is training the single branch network with 1.4k strong and 1.5k weak data. In the original paper, they used 9k weak data but since we have limited computing power in Google Colab, we only trained the network with 1.5k weak data which we obtained from SBD dataset via feeding it to DSRG as a weakly annotated data generation method. In the paper they got 62.8 mIoU as the performance outcome for 9k weak data case in addition to 1.4k strong data. In our case, we obtained 28.1 which is an expected result considering that we had much less weak data compared to the original work. 

![image](https://user-images.githubusercontent.com/85442860/127392950-fd455836-bf3b-4d48-90b3-114a6158074e.png)

Figure 6. Loss curve for Single Branch Network trained with strong and weak annotations

### 3.1.3. Experiments with the Dual Branch Network

We performed an experiment with the dual branch network by feeding 1.4k strongly annotated data to the strong branch and 1.4k weakly annotated data to the weak branch. We could train the network for 20 epochs with considerably less amount of data (ten percent of data) when compared to the original paper.   

![image](https://user-images.githubusercontent.com/85442860/127393073-dd914627-fc69-4a14-87f9-506abe5e7082.png)

Figure 7. Loss curve for Dual Branch Network trained with strong and weak annotations (The first zero value at 0 is just because of initialization for print purposes) 

## 3.2. Running the code

We carried out all implementations on the Google Colab environment. Therefore, some syntactic properties are adjusted to the environment. Our implementation consists of 4 Colab files:
1.	Datasets.ipynb: Contains class definitions for VOC, weak and dual branch training datasets.
2.	Single_Branch.ipynb: Contains network definition, training and prediction functions for carrying out single branch experiments. There are two training functions, one for training with only strong data. The other is for training with both strong and weak data. Imports Datasets.ipynb for using the Dataset definitions.
3.	Dual_Branch.ipynb: Contains network definition, training and prediction functions for the dual branch network setup. Imports Datasets.ipynb. 
4.	Evaluate.ipynb: Contains necessary definitions and functions for evaluating predictions made as outcomes of experiments with Single Branch and Dual Branch networks.
The Colab files contain comments and are self explanatory. Interested readers are invited to refer to the Colab files for implementation details.
While running our files, we made extensive use of Google drive. We usually mounted Google drive to get rid of the time for downloading datasets. As a preliminary step, we downloaded  some files and merged with other files and then stored on our Google drive accounts to be able to use them in code. Thus, before  some of the steps in the implementation phases, we recommend these datasets and related files to be downloaded, prepared and stored on an external place like Google drive. In the rest of this section, the directory structure and data preparation steps are explained. 
1.	Preparing VOC dataset file: Download the dataset (!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and copy the tar file to your Google drive account as VOC_strong.tar 
2.	Preparing Weak dataset file: Download SBD from [4] and then open up with the comment:
	tar -xzf /content/drive/MyDrive/SBD_Strong.tgz
To get train_noval.txt that contains the names of image files in the dataset and prepare a tgz file on drive, run the commands below:

	!wget http://home.bharathh.info/pubs/codes/SBD/train_noval.txt  

	!mv train_noval.txt benchmark_RELEASE

	!tar -cvzf /content/drive/MyDrive/SBD_Strong.tgz benchmark_RELEASE

3.	Generating weak annotations: After predicting with DSRG (as explained in Section 3.1.1), store weakly annotated data generated from SBD image set on drive in directory SBDWeak. The weak annotations reside in SBDWeak/1/pred.  
4.	When VOC_Strong.tar file is extracted, it generates a directory named “VOCdevkit/VOC2012” at the local directory. VOCdevkit/VOC2012/JPEGImages contains input image files in JPEG format. VOCdevkit/VOC2012/SegmentationClass stores the ground truth information, strongly annotated image files in PNG format.
5.	 When SBD_Strong.tgz is extracted, it generated a directory named “benchmark_RELEASE” at the local directory. benchmark_RELEASE/dataset/img contains input images in JPEG format. train_noval.txt contains names of the image files to be loaded.
6.	Preparing pre-encoded label mask files: Both for strongly and weakly annotated data, we need label mask information of the annotations to be able to use them in pixel-wise cross-entropy loss. Label mask information stores the class value corresponding to each pixel in the data. The DataSet classes are capable of generating these pre-encoded mask labels if they are created with generate_pre_encoded=True parameter. For VOC strong dataset, the pre-encoded files are generated in VOCdevkit/VOC2012/SegmentationClass/pre_encoded. To be able to reuse these files in training, we archived and stored them on Google drive and did not re-generate every time we created the dataset. The file names are VOCStrong-pre_encoded.tar and SBDWeak-pre_encoded.tar respectively. The pre-encoded files are in benchmark_RELEASE/weak_pre_encoded for weak annotations.
7.	At each five epoch, we store a checkpoint file on the local directory. The training information (checkpoint files) are stored in SingleBranchStrongData/Training/, SingleBranchStrongWeakData/Training, and DualBranch/Training directories respectively for each kind of training performed in the scope of this study.
8.	The predictions are stored under SingleBranchStrongData/Results/pred, SingleBranchStrongWeakData/Results/pred and DualBranch/Results/pred local directories as .PNG files.
9.	Evaluation results are stored in text files under SingleBranchStrongData/Results/, SingleBranchStrongWeakData/Results/ and DualBranch/Results/. The file name is “evaluation_out.txt”. 

## 3.3. Results

As in the paper, our metric for evaluation is meanIoU. In the paper, performance of the dual branch network is compared with the performance of DSRG method. Because of this reason, we made use of the same evaluation metric as DSRG. We adapted the evaluation codes of DSRG implementation [7]  (main/evaluation.py) to evaluate the performance of the models so that we can compare the results with the DSRG outcomes.
When we evaluated the predictions of the DSRG network we trained for 1500 epochs (the recommended number of epochs were 8000), the overall mIoU for the predictions of the model is 42.71. This value is a good baseline for evaluating the performance of our models.  Table 1 shows the mIoU values for each of the classes in the PASCAL VOC dataset.

![image](https://user-images.githubusercontent.com/85442860/127393310-95b3c812-b07a-449a-8a73-b987a2642b2e.png)

Table 1. Evaluation results for the weakly generated dataset for the DSRG implementation

### 3.3.1. Single Branch with strong annotated data

![image](https://user-images.githubusercontent.com/85442860/127393510-f337c98b-1126-41c8-99d8-ea0a798fb87e.png)

Table 2. Evaluation results for the Single Branch Network trained with strongly annotated dataset

![image](https://user-images.githubusercontent.com/85442860/127393650-9708afea-4a66-4c78-9a45-955707acdc15.png)

Figure 8. Prediction from the Single Branch Network trained with strongly annotated data. The input image is shown on the left.

### 3.3.2. Single Branch with strong+weakly annotated data

![image](https://user-images.githubusercontent.com/85442860/127393848-bab45b89-6b55-40bf-9322-89ee233c3f7c.png)

Table 3. Evaluation results for the Single Branch Network trained with strong and weakly annotated dataset

![image](https://user-images.githubusercontent.com/85442860/127393939-8478a53b-635a-45f7-92c5-dce652b33cc9.png)

Figure 9. Prediction from the Single Branch Network trained with strong+weakly annotated data. The input image is shown on the left.

### 3.3.3. Dual Branch with strong+weakly annotated data

Since we had a limited GPU, we trained the model for 20 epochs and although training loss values are satisfying we couldn’t obtain good predictions for the dual branch case. mIoU was 3.3, the predictions were random figures as shown below. 

![image](https://user-images.githubusercontent.com/85442860/127394043-d853aced-b3d2-4509-aad0-dbeb6def9416.png)

Figure 10. Prediction from the Dual Branch Network trained with strong+weakly annotated data. The input image is same with previous figure.

# 4. Conclusion

As we mentioned, we implemented a part of the original paper because of the time and resource constraints. Considering the reproduced part, we can say that our results are not aligned with the original paper in a sense that we observed that enriching limited amount of strongly annotated data with much more easily obtainable and cheap weakly annotated data improved performance. However, we have some interpretation about this outcome, which is provided in detail in the following paragraphs. Besides, paper suggests that apart from data enrichment network structure should be changed as well to improve the performance. However, although we implemented and trained our dual branch network with strong and weak data, we did not get good prediction results. Thus we could not observe that “improving the segmentation performance is possible with enriching the small amount of strongly (pixel-wise) annotated data via using a dual branch network”. The reason behind this result is also discussed below.

In the study, one of the main points that authors mention a few times is the lack of increase in performance while using large sized weakly annotated data besides the small sized strongly annotated data. They also pointed out an interesting outcome, the performance is degraded while using weak data with strong data. Thus, in contrast to its objective weak data usage did not strengthen the performance it even decreased it. However, in our experiments we observed a different outcome. In single branch experiments, with only strong data we obtained approximately 22 meanIoU but with weak and strong data together, we observed an increase of approximately 6 in meanIoU and we obtained 28. This is not aligned with the paper’s outcomes since in our experiments weak data actually helped our results when combined with strong data. On the other hand, we used approximately same amount of weakly annotated data with strong data but in the paper they used really big amount of weak data (9k) compared to strong data (1.4k) and according to their second figure which we show as figure 1, they concluded that weak data is not helping to improve performance after many (totally 150) epochs. In that graph, we see that for 20 epochs the performance of combined (weak and strong data) case is higher than the only strong data case. In this regard we concluded that our results are not unexpected considering our training which is 20 epochs. Another point we want to mention is that, we could not fully replicate the training duration, we could not train the model for 150 epochs since we had a very limited GPU which is provided by colab and just for a really short time. 

In dual branch experiment, authors again used a large amount of weakly annotated data compared to strong data (9k vs 1.4k) but in our case, we just had chance to use same amount of strong and weak data (1.4 k approximately each), since we had limited training capacity we could not feed the network with large data. In dual branch experiment, our training performance was good and our loss curve was decreasing. However, in prediction we did not obtain good results in test set. Actually, we obtained really low results. There are some reasons behind this failure. One reason is that since we had limited GPU we could not train the network for a long time, we trained for 20 epochs. Another reason is that we were not able to use great number of weak data such as 9k in the paper, we just utilized from approximately 1.4k strongly and weakly annotated images. Thus, we can say that although we established a dual branch network and managed to train it we could not obtain successful prediction results because of resource scarcity. 

To sum up, we also want to mention some difficulties and constraints which put some pressure on us and lie behind our results. Since we used Google Colab we faced with GPU allocation constraints most of the time. Google Colab is a very useful and interactive environment for Python implementation, however it is a freely provided resource all over the world so the resource limitation is inevitable. But  in deep learning, since we need an uninterrupted GPU allocation for an adequate time, especially trainings were challenging for us. Yet we have a gratitude since we were able to use such a good resource throughout our project. Lastly we want to add that this project  was very helpful for us understanding many concepts more clearly via implementing them by ourselves.


# 5. References

[1]  Wenfeng L., Meng Y., Semi-supervised Semantic Segmentation via Strong-weak Dual-branch Network, ECCV 2020,
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500766-supp.pdf 
[2] Huang et. al., Weakly-Supervised Semantic Semantic Segmentation Network with Deep Seeded Region Growing, CVPR 2018, http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf 
[3] PASCAL VOC 2012 Dataset: 
https://deepai.org/dataset/pascal-voc, https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/PascalVOC_IJCV2009.pdf 
[4] Semantic Boundaries Dataset: http://home.bharathh.info/pubs/codes/SBD/download.html
[5] DSRG PyTorch implementation: https://github.com/terenceylchow124/DSRG_PyTorch/
[6] Main source for the dataset implementations:  https://github.com/DevikalyanDas/Semantic-Segmentation-of-pascal-voc-dataset-using-Pytorch/blob/main/Pascal_VOC_Segmentation.ipynb  
[7] Our DSRG fork (originally taken from https://github.com/terenceylchow124/DSRG_PyTorch and revised for our project) for training DSRG and generating the weak dataset is available as a separate zip file named "DSRG_PyTorch-main"


# Contact

Duygu Dinç,duyguddinc@gmail.com	  Evren Çilden, evren.cilden@gmail.com

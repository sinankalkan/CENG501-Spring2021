
# Meticulous Object Segmentation

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

[This paper[1]](https://arxiv.org/pdf/2012.07181.pdf) was published by Adobe as an Arxiv paper in 2020. The paper proposes a solution which is called MeticulousNet for the segmentation of well defined foreground objects including high resolution images. They also called this task as "Meticulous Object Segmentation". The results shown in the article are highly impressive both qualitatively and quantitatively. MeticulousNet consists of two models which have almost same architecture. These models are MOS_L and MOS_H that are responsible from the coarse(low resolution) and refined(high resolution) mask generation respectively.

I aimed to obtain similar qualitative and quantitative results with the paper for both MOS_L and MOS_H.

## 1.1. Paper summary

The paper aims to generate coarse and fine mask of the major foreground object. Coarse mask generation is done just like salient object segmentation. The MOS_L model is responsible for coarse mask generation. The MOS_H model is used to improve the coarse mask produced from the MOS_L model. MOS_L and MOS_H models have the same architecture, but there is only one difference between them, which is that MOS_L using RGB image as input whereas MOS_H using RGB image and coarse mask as input.

The MOS_L model is trained with the [DUTS-TR[2]](http://saliencydetection.net/duts/) dataset. Inputs are resized to 336x336 and with this resolution a coarse mask is obtained in both training and testing phases. 

The MOS_H model is trained with the MSRA-10K [3], DUT-OMRON [4], ECSSD [5], and FSS-1000[6]. DIM[7] dataset is also added to MOS_H training dataset with 463 unique foreground. It is synthesized 100 times with random background images sampled from the COCO dataset for each foreground in the DIM dataset. MOS_H refines images in patch-based manner. Even high-resolution images, which are normally difficult to fit into GPU memory, can be used with this patch-based refinement method. At training time, they used 224x224 random cropped images with perturbed corresponding ground truth masks. The method used at the test time is not clearly stated in the article. I have used 224x224 windows to crop test images without overlapping.

<p align="center">
<img src="figures/figure2.png" alt="drawing" width="50%"/>
</p>

# 2. The method and my interpretation

## 2.1. The original method

MeticulousNet has encoder-decoder architecture. Encoder part can be any feature extraction module from other networks. In this paper, encoder of the PSPNet with ResNet50 is used as the encoder. The main contribution of the paper comes from the decoder part of MeticulousNet.


 

## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

Experiemental setup is not defined in the paper. I have used RTX2080Ti to train network. In the paper they have expressed that their both MOS_L and MOS_H models are trained 60000 iterations with 20 and 24 batch sizes respectively. Since GPU memory can not fit these batch sizes, I have used 4 and 8 batch sizes for MOS_L and MOS_H respectively. I have trained models with ~30000 iterations which less than mentioned in the paper.

## 3.2. Running the code
#### Preparation	
* First download the training datasets by using the [download_training_datasets.py](/scripts/download_training_datasets.py) in the scripts folder.
* Set up your environment using a [requirements.txt](requirements.txt)

#### Folder Structure
* The "Configs" folder contains the necessary hyperparameters and settings for training.
* Download training datasets into the datasets folder. Change the path of the datasets in the config files.

#### Training

```
python train.py --config ./configs/MOS_L.json --device cuda
```

```
python train.py --config ./configs/MOS_H.json --device cuda
```


<i>To continue from the saved checkpoint you can use resume argument:</i>

```
python train.py --config ./configs/MOS_L.json --resume ./saved/MeticulousNet_H/07-28_05-32/checkpoints/checkpoint-epoch30.pth --device cuda
```

#### Test

<b> Coarse Mask Generation: </b>

```
python /lib/eval/coarse.py
```

<b> Fine Mask Generation: </b>

```
python /lib/eval/fine.py
```

<b> Calculate Evaluation Metrics: </b>

```
python /lib/eval/eval_post.py --gt-dir ./dataset/test/mask --pred-dir ./saved/outputs/test/mask
```

## 3.3. Results

### Quantitative Results:
#### Mos_L Results:

|               | IoU   | mBA   |
|---------------|-------|-------|
| Paper Results | 78.40 | 66.42 |
| My Results    | 70.04 | 63.42 |

Table 1: Low resolution foreground segmentation on DUTS-TE


|               | IoU   | mBA   |
|---------------|-------|-------|
| Paper Results | 80.59 | 66.09 |
| My Results    | 74.87 | 63.65 |

Table 2: Low resolution foreground segmentation on HRSOD.

#### Mos_H Results:

|               | IoU   | mBA   |
|---------------|-------|-------|
| Paper Results | 82.56 | 73.64 |
| My Results    | 74.60 | 67.14 |

Table 3: High resolution foreground segmentation on HRSOD.

### Qualitative Results:
#### MOS600 Subset Outputs:
<table>
  <tr>
  <td style="word-wrap: break-word">
      My MOS_L Outputs: </td>
    <td> <img src="saved/outputs/MOS600/coarse/sample1.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/MOS600/coarse/sample2.jpg"alt="2" height=240px></td>
    <td> <img src="saved/outputs/MOS600/coarse/sample3.jpg"alt="3" height=240px></td>
    <td> <img src="saved/outputs/MOS600/coarse/sample4.jpg"alt="4" height=240px></td>
    <td> <img src="saved/outputs/MOS600/coarse/sample5.jpg"alt="5" height=240px></td>
    <td> <img src="saved/outputs/MOS600/coarse/sample6.jpg"alt="6" height=240px></td>
   </tr> 
   
  <tr>
  <td style="word-wrap: break-word">
      My MOS_H Outputs: </td>
    <td> <img src="saved/outputs/MOS600/fine/sample1.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/MOS600/fine/sample2.jpg"alt="2" height=240px></td>
    <td> <img src="saved/outputs/MOS600/fine/sample3.jpg"alt="3" height=240px></td>
    <td> <img src="saved/outputs/MOS600/fine/sample4.jpg"alt="4" height=240px></td>
    <td> <img src="saved/outputs/MOS600/fine/sample5.jpg"alt="5" height=240px></td>
    <td> <img src="saved/outputs/MOS600/fine/sample6.jpg"alt="6" height=240px></td>
   </tr> 
   
  <td style="word-wrap: break-word">
      Paper MOS_H Outputs: </td>
    <td> <img src="saved/outputs/MOS600/gt/sample1_mask.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/MOS600/gt/sample2_mask.jpg"alt="2" height=240px></td>
    <td> <img src="saved/outputs/MOS600/gt/sample3_mask.jpg"alt="3" height=240px></td>
    <td> <img src="saved/outputs/MOS600/gt/sample4_mask.jpg"alt="4" height=240px></td>
    <td> <img src="saved/outputs/MOS600/gt/sample5_mask.jpg"alt="5" height=240px></td>
    <td> <img src="saved/outputs/MOS600/gt/sample6_mask.jpg"alt="6" height=240px></td>
   </tr> 
</table>

#### HRSOD-Test Subset Outputs:

<table>
  <tr>
  <td style="word-wrap: break-word">
      My MOS_H Outputs on HRSOD: </td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/1342827_a898640c38_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/4923889_ee3a448ece_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/5242449_0157834540_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/49026382_ea219c9aeb_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/147383949_ad0022dbed_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/187179248_103d2db5ef_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/255993064_3547a828e0_o.jpg"alt="1" height=240px></td>
    <td> <img src="saved/outputs/HRSOD_Subset/fine/255996486_5d9c90314d_o.jpg"alt="1" height=240px></td>
   </tr> 
</table>
# 4. Conclusion

While the method produces qualitatively impressive results, its quantitative results seem very close to those of SOTA results. This may be one reason why this article has not yet been published. It is very difficult to produce such clear and accurate masks on high resolution images, due to the requirement to resize images to fit the GPU. Thanks to the patch-based refinement method used in this study, they were able to overcome this difficulty. Their architecture looks like a good combination of CascadePSP[8] and Pointrend[9].

# 5. References

1. Yang, Chenglin, et al. "Meticulous Object Segmentation." _arXiv preprint arXiv:2012.07181_ (2020).
2. Wang, Lijun, et al. "Learning to detect salient objects with image-level supervision." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2017.
3. Cheng, Ming-Ming, et al. "Global contrast based salient region detection." _IEEE transactions on pattern analysis and machine intelligence_ 37.3 (2014): 569-582.
4. Yang, Chuan, et al. "Saliency detection via graph-based manifold ranking." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2013.
5. Shi, Jianping, et al. "Hierarchical image saliency detection on extended CSSD." _IEEE transactions on pattern analysis and machine intelligence_ 38.4 (2015): 717-729.
6. Li, Xiang, et al. "Fss-1000: A 1000-class dataset for few-shot segmentation." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2020.
7. Xu, Ning, et al. "Deep image matting." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2017.
8. Cheng, Ho Kei, et al. "CascadePSP: toward class-agnostic and very high-resolution segmentation via global and local refinement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
9. Kirillov, Alexander, et al. "Pointrend: Image segmentation as rendering." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

# Contact

Onur Can ÜNER - onurcan.uner@gmail.com

# Momentum Batch Normalization for Deep Learning with Small Batch Size

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

<!--- Introduce the paper (inc. where it is published) and describe your goal (reproducibility).-->
When the training batch size is small, the noise level becomes high, increasing the training difficulty.\
To overcome this difficulty,Hongwei Yong proposed "Momentum Batch Normalization for Deep Learning with Small Batch Size" in 2020 and it was published at European Conference on Computer Vision (ECCV).\
Our aim is to implement the models described in the paper and obtain the acceptable results.

## 1.1. Paper summary

Summarize the paper, the method & its contributions in relation with the existing literature.

Computing power in the last decade especially GPUs processing capacities increased remarkably and this improvement directly affected the deep neural networks (DNNs) success.  DNNs are used in various applications; object detection, natural language processing, speech recognition etc. The main secret behind the success of DNNs lies behind the usage of large scale datasets, learning algorithms and considerably higher computational power.

Normalizing the data and intermediate features in DNNs is one of the key fetaures which affects the success of the architecture. Normalizing the input data improves the training performance and resulting lower training times as well. In literature it is quite common to use Batch Normalization (BN) technique in order to normalize the samples of mini-batches during the training process. It is a well known fact that BN improves training speed and performance. It also allows user to choose higher learning rates while it improves the generalization capacity of the model.

It is not quite certain how BN improves the performance yet, it is argued that BN reduce the Internal Covariance Shift (ICS) but xxxx opposes that there is not an evidence between ICS and BN. However, it is certain that BN adds some noise. In the original paper authors propose a new method which is called Momentum Batch Normalization (MBN). According to the authors the added noise to the mean and variance depends on only the batch-size in conventional BN technique. Smaller batch-sizes increases the noise and this phenomena makes the training more difficult. MBN method can automatically controls the noise level in the training process and this allows a stable training even with smaller batch-sizes when the memory resources are insufficient. 


# 2. The method and my interpretation

## 2.1. The original method

<!--- Explain the original method.\ -->

In the original method authors propose a dynamic momentum parameter <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> in order to control the final noise level in the training. The algorithm of the MBN which is directly taken from the paper is given below;

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\text&space;{&space;Training&space;step:&space;}\\&space;&\begin{aligned}&space;&\mu_{B}=\frac{1}{M^{2}}&space;\sum_{i=1}^{m}&space;x_{i}&space;\\&space;&\sigma_{B}^{2}=\frac{1}{m}&space;\sum_{i=1}^{m}\left(x_{i}-\mu_{b}\right)^{2}&space;\\&space;&\mu&space;\leftarrow&space;\lambda&space;\mu&plus;(1-\lambda)&space;\mu_{B}&space;\\&space;&\sigma^{2}&space;\leftarrow&space;\lambda&space;\sigma^{2}&plus;(1-\lambda)&space;\sigma_{B}^{2}&space;\\&space;&\widehat{x}_{i}=\frac{x_{i}-\mu}{\sqrt{\sigma^{2}&plus;e}}&space;\\&space;&y_{i}=\gamma&space;\widehat{x}_{i}&plus;\beta&space;\\&space;&\mu_{i&space;n&space;f}&space;\leftarrow&space;\tau&space;\mu_{i&space;n&space;f}&plus;(1-\tau)&space;\mu_{B}&space;\\&space;&\sigma_{i&space;n&space;f}^{2}&space;\leftarrow&space;\tau&space;\sigma_{i&space;n&space;f}^{2}&plus;(1-\tau)&space;\sigma_{B}^{2}&space;\\&space;&\text&space;{&space;Inference&space;step:&space;}&space;y_{i}=\gamma&space;\frac{x_{i}-\mu_{i&space;n&space;f}}{\sqrt{\sigma_{i&space;n&space;f}^{2}&plus;e}}&plus;\beta&space;\end{aligned}&space;\end{aligned}" title="\begin{aligned} &\text { Training step: }\\ &\begin{aligned} &\mu_{B}=\frac{1}{M^{2}} \sum_{i=1}^{m} x_{i} \\ &\sigma_{B}^{2}=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{b}\right)^{2} \\ &\mu \leftarrow \lambda \mu+(1-\lambda) \mu_{B} \\ &\sigma^{2} \leftarrow \lambda \sigma^{2}+(1-\lambda) \sigma_{B}^{2} \\ &\widehat{x}_{i}=\frac{x_{i}-\mu}{\sqrt{\sigma^{2}+e}} \\ &y_{i}=\gamma \widehat{x}_{i}+\beta \\ &\mu_{i n f} \leftarrow \tau \mu_{i n f}+(1-\tau) \mu_{B} \\ &\sigma_{i n f}^{2} \leftarrow \tau \sigma_{i n f}^{2}+(1-\tau) \sigma_{B}^{2} \\ &\text { Inference step: } y_{i}=\gamma \frac{x_{i}-\mu_{i n f}}{\sqrt{\sigma_{i n f}^{2}+e}}+\beta \end{aligned} \end{aligned}" />

The formulation of the momentum parameter of the MBN for the training stage is as follows;

<img src="https://latex.codecogs.com/gif.latex?\lambda^{(t)}=\rho^{\frac{T}{T-1}&space;\max&space;(T-t,&space;0)}-\rho^{T},&space;\quad&space;\rho=\min&space;\left(\frac{m}{m_{0}},&space;1\right)^{\frac{1}{T}}" title="\lambda^{(t)}=\rho^{\frac{T}{T-1} \max (T-t, 0)}-\rho^{T}, \quad \rho=\min \left(\frac{m}{m_{0}}, 1\right)^{\frac{1}{T}}" />

where <img src="https://latex.codecogs.com/gif.latex?t" title="t" /> refers to the <img src="https://latex.codecogs.com/gif.latex?t" title="t" /> -th iteration epoch, <img src="https://latex.codecogs.com/gif.latex?T" title="T" /> is the number of the total epochs. As it is seen from the formulation above in the beginning phase of the training the normalization technique does not differ much from the conventional Batch Normalization but as the epoch number increase the momentum parameter shows its effect. The momentum parameter <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> does not change within one epoch.

The formulation of the momentum parameter of the MBN for the inference stage is as follows;

<img src="https://latex.codecogs.com/gif.latex?\tau^{\frac{N}{m}}=\tau_{0}^{\frac{N}{m_{0}}}&space;\Rightarrow&space;\tau=\tau_{0}^{\frac{m}{m_{0}}}" title="\tau^{\frac{N}{m}}=\tau_{0}^{\frac{N}{m_{0}}} \Rightarrow \tau=\tau_{0}^{\frac{m}{m_{0}}}" />

The inference momentum parameter is adaptive to batch size. Where <img src="https://latex.codecogs.com/gif.latex?m" title="N" /> denotes the number of samples in other words batch size and <img src="https://latex.codecogs.com/gif.latex?m_{0}" title="m_{0}" /> denotes the ideal batch size which is set to 32 in the original paper and it is preserved in the experiments of this implementation as well. In addition <img src="https://latex.codecogs.com/gif.latex?{\tau&space;_{0}}" title="{\tau _{0}}" /> is an ideal coefficient and in the originial paper authors set it to 0.90 for the inference step. According to the inference step momentum formula, it can be easily seen that smaller batch size makes the inference momentum parameter larger. Therefore, the noise in momentum mean and variance will be kept under control.


Some implementation details are given in the paper.\
•	Learning rate: It starts with 0.1*m/64 for both datasets and divide it by 10 for every 60 epochs.\
•	Ideal batch size, <img src="https://latex.codecogs.com/gif.latex?m_{0}" title="m_{0}" /> :32 \
•	Batch size, <img src="https://latex.codecogs.com/gif.latex?m" title="m" /> : 8-16-32 \
•	Ideal inference momentum parameter, <img src="https://latex.codecogs.com/gif.latex?{\tau&space;_{0}}" title="{\tau _{0}}" />  : 0.9 \
•	Total epoch number, <img src="https://latex.codecogs.com/gif.latex?T" title="T" />: 200 \
•	Neural network model : RESNET18,RESNET34,RESNET50,VGG11,VGG16 \
•	Dataset: CIFAR10 and CIFAR100 \
•	Optimizer: SGD 

## 2.2. My interpretation 

<!--- Explain the parts that were not clearly explained in the original paper and how you interpreted them.-->
Some implementation details are not given in the paper. Therefore, we interpreted them based on literature. \
•	Data Augmentation\
•	Loss Function

Data augmentation is not clearly explained in the paper. In the begining of the experiments there were a gap between our and authors results. Therefore, data augmentation is implemented according to the procedure from the literature [[2]](#2). When it is applied to datasets in our experiments results became closer to the original paper's results so in every experiments conducted in this project this method is used. 

It is also not quite certain in the paper that which loss function is used in the experiments. Firstly, the data augmentation method is fixed and the Cross-Entropy Loss function is used in ResNet18 model with CIFAR10 dataset in our experiment. Then the results are compared with the original experiment and it was observed that the obtained results are compatible with the original paper. Therefore, it is assumed that Cross-Entropy Loss function would be reasonable for this problem and in all of the experiments it is used as our loss function. 


# 3. Experiments and results

## 3.1. Experimental setup

In this paper, to evaluate MBN, this method is applied to image classification tasks and experiments are conducted on CIFAR10,CIFAR100 and  Mini-ImageNet100 datasets.\
The computation time without disconnection is limited on Google Colab.To solve this problem, we used Colab Pro. With Colab Pro, we get priority access to runtime up to 24 hours and twice as many CPUs as standard Colab.  However, Google still applies usage limits on GPU, forcing us to be selective about the runs to have proceeded.Therefore ,we decided to reproduce results for CIFAR10 and CIFAR100. Since, the obtained results with CIFAR10 and CIFAR100 datasets showed almost the same trend with the original paper, Mini-ImageNet-100 dataset is not implemented in our experiments. 

The CIFAR10 and CIFAR100 datasets were loaded using torchvision.datasets. Other experimental parameters and details are given in Section 2. Batch size is changed in different experiments by keeping it compatible with the original paper to reproduce the results of the original paper. 

In the original paper authors used four GPUs for the experiments and they splitted the batches by four. Therefore, the results in the original paper is given according to the batch size per GPU. However, throughout this implementaion project only one GPU is used because of insufficient computational resources. Therefore, in our figures, results etc. batch size is shown as multiplied by four because only one GPU is used as mentioned. 

## 3.2. Running the code

The notebooks are written for running on Google Colab. All dependencies are available within the current runtime on Google Colab.\
There are ten code notebooks- five for each dataset- one is for MBN and the other is for BN, Layer Normalization (LN), Group Normalization (GN) and Instance Normalization (IN). Each notebook involves all the models such as Resnet18, Resnet34, Resnet50, VGG11 and VGG16. Depending on the experiment, the others should be commented out. Each notebook can be run using "Run All".

## 3.3. Results

It can be seen in Figure-4 that MBN gives similar results with the BN in larger batch-sizes. In fact, in the obtained results it is observed that BN shows slightly better performance compared to MBN such as batch size 16 and 32. It is similar in the original paper as well. Even though with the batch size 32 the results are not similar with the original paper, BN performs better in batch size 16 in the original paper. However, as expected MBN shows a better performance with the smallest batch size. This results shows that MBN provides a more robust training with smaller batch sizes compared to other normalization methods. The main idea behind the method proposed by the authors; MBN suppresses the noise in the mean and variance in smaller batch sizes. Moreover, our results can be accepted as compatible with the original paper, because MBN shows better performance in the smallest batch size compared to other normalization methods. In addition, there is a increasing performance trend in favor to the MBN while batch size decreases.

| ![fig4.jpg](https://user-images.githubusercontent.com/86877356/127033391-a8138ded-9496-4c7a-8371-ed23881eeb90.png) | 
|:--:| 
| *Fig 4: Testing accuracy on CIFAR10 and CIFAR100 of ResNet18 with training batch size (BS) 8 and 32.* |

In Figure-5 it can be seen that the training and testing accuracy curves vs. epoch of ResNet18 with batch size 8. In the original paper, especially in CIFAR100 dataset, in the last section of the training MBN can carry out a considerable performance gain compared to other normalization methods. In our results, it could not be observed such a dominant performance difference in favor to the MBN. However, in both of the datasets MBN shows a better performance and in the last epochs of the training it maintains to increase the accuracy level. This phenomena is consistent with the method promises. 

| ![fig5.jpg](https://user-images.githubusercontent.com/86877356/126895343-f17f62e0-db71-4999-aa6b-60b924377ef0.png) | 
|:--:| 
| *Fig 5: Comparison of accuracy curves for different normalization methods with  batch size of 8.We show the test accuracies vs. the epoches  on CIFAR10(top) and CIFAR100(bottom). The ResNet is used.* |



In Figure-6 (a) it can be seen that the training and testing accuracy curves vs. epoch and Figure-6 (b) final testing accuracies. It can be seen in our results which are consistent with the original paper, MBN shows better performance in all those different networks. In ResNet and VGG when the total layer of the network increases of course the total number of normalization layers increase. This increase in normalization layers consequences the increase of the added noise. In other words, when the network is getting deeper the accumulated noise level shows an increase. With the smaller batch size this effec can be seen better. 

In our results, this effect can be seen in the Figure-6 (a); in the beginning stage of the training process MBN could not handle the accumluated noise level but in the last stage of the training owing to dynamic momentum parameter MBN starts to suppress the noise level and it maintains the performance gain better compared to other normalization method. It is seen in batch size 8 results more clearly.


 We
can have the following observations. First, on all the four networks, MBN always
outperforms BN. 

Second, under such a small batch size, the accuracy of deeper
network ResNet50 can be lower than its shallower counterpart ResNet34. That
is because the deeper networks have more BN layers, and each BN layer would
introduce relatively large noise when batch size is small. The noise is accumu-
lated so that the benet of more layers can be diluted by the accumulated noise.
However, with MBN the performance drop from ResNet50 to ResNet34 is very
minor, where the drop by BN is significant. This again validates that MBN can
suppress the noise eectively in training.


| ![fig6a.jpg](https://user-images.githubusercontent.com/86877356/127190495-a1a7d4a6-46ad-44c2-91b7-d5fbcf48c320.jpg) | 
|:--:| 
| *Fig 6a: Testing accuracy curves on CIFAR100 for different network architectures with training batch size of 8 and 16* |


| ![fig6b.jpg](https://user-images.githubusercontent.com/86877356/126891348-60b4945f-9fef-4f1c-841a-66931383f8ec.png) | 
|:--:| 
| *Fig 6b: Testing accuracy on CIFAR100 for different network architectures with training batch size of 8* |

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

<a name="2"></a>

### <a name="tith"></a>[2] https://medium.com/swlh/how-data-augmentation-improves-your-cnn-performance-an-experiment-in-pytorch-and-torchvision-e5fb36d038fb





https://blog.paperspace.com/pytorch-101-building-neural-networks/

# Contact

Provide your names & email addresses and any other info with which people can contact you.\
Hüseyin Avni Yasar : [@avniyasar](https://github.com/avniyasar) \
Safa Mesut Bostancı : [@smbostanci](https://github.com/smbostanci)

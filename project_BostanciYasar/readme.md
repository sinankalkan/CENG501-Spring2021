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
Some implementation details are not given in the paper.Therefore, we interpreted them based on literature. \
•	Data Augmentation\
•	Preprocessing Techniques\
•	Loss Function

# 3. Experiments and results

## 3.1. Experimental setup

<!--- Describe the setup of the original paper and whether you changed any settings.
With Colab Pro you get priority access to high-memory VMs. These VMs generally have double the memory of standard Colab VMs, and twice as many CPUs. You will be able to access a notebook setting to enable high-memory VMs once you are subscribed.-->
<!--- The computation time without disconnection is limited on Google Colab.To solve this problem,we used Colab Pro.With Colab Pro,we get priority access to runtime up to 24 hours and twice as many CPUs compared to standard Colab.However,virtual resources are still limited due to Colab policy.However,Google still apply usage limits on GPU and this forces us to be selective about the runs to be proceeded-->
<!---  Google needs to maintain the flexibility to adjust usage limits and the availability of hardware on the fly. -->
In this paper, to evaluate MBN,this method is applied to image classification tasks and experiments are conducted on CIFAR10,CIFAR100 and  Mini-ImageNet100 datasets.\
The computation time without disconnection is limited on Google Colab.To solve this problem, we used Colab Pro. With Colab Pro, we get priority access to runtime up to 24 hours and twice as many CPUs as standard Colab.  However, Google still applies usage limits on GPU, forcing us to be selective about the runs to have proceeded.Therefore ,we decided to reproduce results for CIFAR10 and CIFAR100.
The CIFAR10 and CIFAR100 datasets were loaded using torchvision.datasets. Other experimental parameters and details are given in Section 2. 
## 3.2. Running the code

Explain your code & directory structure and how other people can run it.

The notebooks are written for running on Google Colab. All dependencies are available within the current runtime on Google Colab.\
There are four code notebooks- two for each dataset- one is for MBN and the other is for BN.Each notebook involves all the models.Depending on the experiment,the others should be commented out. Each notebook can be run using "Run All", but the run time may be excessive to complete it all in one run. One might suggest deciding number of epochs based on the available runtime limitations.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

<!---
| ![fig4.jpg](https://user-images.githubusercontent.com/86877356/126894727-e119b037-d039-4173-b9de-58625b7304e8.png) | 
|:--:| 
| *Fig 4: Testing accuracy on CIFAR10 and CIFAR100 of ResNet18 with training batch size (BS) 8 and 32.* |
-->

| ![fig4.jpg](https://user-images.githubusercontent.com/86877356/127033391-a8138ded-9496-4c7a-8371-ed23881eeb90.png) | 
|:--:| 
| *Fig 4: Testing accuracy on CIFAR10 and CIFAR100 of ResNet18 with training batch size (BS) 8 and 32.* |



| ![fig5.jpg](https://user-images.githubusercontent.com/86877356/126895343-f17f62e0-db71-4999-aa6b-60b924377ef0.png) | 
|:--:| 
| *Fig 5: Comparison of accuracy curves for different normalization methods with  batch size of 8.We show the test accuracies vs. the epoches  on CIFAR10(top) and CIFAR100(bottom). The ResNet is used.* |






| ![fig6a.jpg](https://user-images.githubusercontent.com/86877356/127190495-a1a7d4a6-46ad-44c2-91b7-d5fbcf48c320.jpg) | 
|:--:| 
| *Fig 6a: Testing accuracy curves on CIFAR100 for different network architectures with training batch size of 8 and 16* |

<!---
| ![fig6a.jpg](https://user-images.githubusercontent.com/86877356/127048299-710a1bde-9398-4811-af21-8b5a34cc69e5.png) | 
|:--:| 
| *Fig 6a: Testing accuracy curves on CIFAR100 for different network architectures with training batch size of 8 and 16* |


| ![fig6a.jpg](https://user-images.githubusercontent.com/86877356/127045192-7586b030-8608-4637-a3aa-3b2aace0fe3c.png) | 
|:--:| 
| *Fig 6a: Testing accuracy curves on CIFAR100 for different network architectures with training batch size of 8 and 16* |


| ![fig6a.jpg](https://user-images.githubusercontent.com/86877356/126895471-b42fa9e4-ebe0-4f76-9575-c17fc4c99d3e.png) | 
|:--:| 
| *Fig 6a: Testing accuracy curves on CIFAR100 for different network architectures with training batch size of 8 and 16* |
-->

| ![fig6b.jpg](https://user-images.githubusercontent.com/86877356/126891348-60b4945f-9fef-4f1c-841a-66931383f8ec.png) | 
|:--:| 
| *Fig 6b: Testing accuracy on CIFAR100 for different network architectures with training batch size of 8* |




# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

https://blog.paperspace.com/pytorch-101-building-neural-networks/

# Contact

Provide your names & email addresses and any other info with which people can contact you.\
Hüseyin Avni Yasar : [@avniyasar](https://github.com/avniyasar) \
Safa Mesut Bostancı : [@smbostanci](https://github.com/smbostanci)

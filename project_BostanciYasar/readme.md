# Momentum Batch Normalization for Deep Learning with Small Batch Size

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

<!--- Introduce the paper (inc. where it is published) and describe your goal (reproducibility).-->
When the training batch size is small, the noise level becomes high, increasing the training difficulty.\
To overcome this difficulty,Hongwei Yong proposed "Momentum Batch Normalization for Deep Learning with Small Batch Size" in 2020 and it was published at European Conference on Computer Vision (ECCV).\
Our aim is to implement the models described in the paper and obtain the acceptable results.

## 1.1. Paper summary

Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

<!--- Explain the original method.\ -->
Some implementation details are given in the paper.\
•	Learning rate: It starts with 0.1*m/64 for both datasets and divide it by 10 for every 60 epochs.\
•	Ideal batch size:32 \
•	Batch size: 8-16-32 \
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
The CIFAR10 and CIFAR100 datasets were loaded using torchvision.datasets.
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




| ![fig6a.jpg](https://user-images.githubusercontent.com/86877356/126895471-b42fa9e4-ebe0-4f76-9575-c17fc4c99d3e.png) | 
|:--:| 
| *Fig 6a: Testing accuracy curves on CIFAR100 for different network architectures with training batch size of 8 and 16* |

| ![fig6b.jpg](https://user-images.githubusercontent.com/86877356/126891348-60b4945f-9fef-4f1c-841a-66931383f8ec.png) | 
|:--:| 
| *Fig 6b: Testing accuracy on CIFAR100 for different network architectures with training batch size of 8* |




# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Provide your names & email addresses and any other info with which people can contact you.

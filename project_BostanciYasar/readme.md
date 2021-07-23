# Momentum Batch Normalization for Deep Learning with Small Batch Size

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

Introduce the paper (inc. where it is published) and describe your goal (reproducibility).
When the training batch size is small, the noise level becomes high, increasing the training difficulty.To overcome this difficulty,Hongwei Yong proposed "Momentum Batch Normalization for Deep Learning with Small Batch Size" in 2020 and it was published at European Conference on Computer Vision (ECCV).
Our aim is to implement the models described in the paper and obtain the acceptable results.

## 1.1. Paper summary

Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

Explain the original method.
Some hyperparameters are given in the paper.
•	Learning rate: It starts with 0.1*m/64 for both datasets and divide it by 10 for every 60 epochs.
•	Ideal batch size:32
•	Batch size: 8-16-32
•	Neural network model : RESNET18,RESNET34,RESNET50,VGG11,VGG16
•	Dataset: CIFAR10 and CIFAR100
•	Optimizer: SGD


## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

•	Data Augmentation
•	Preprocessing Techniques
•	Loss Function

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

Provide your names & email addresses and any other info with which people can contact you.

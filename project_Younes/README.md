# Train a One-Million-Way Instance Classifier for Unsupervised Visual Representation Learning

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

This project aims to reproduce the results presented in the in the paper [Train a One-Million-Way Instance Classifier for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/2102.04848.pdf).

## 1.1. Paper summary

The paper presents an unsupervised visual representation learning method  with a pretext task of discriminating all images in a dataset using an instance level classifier. The aim of the paper is to provide novel efficient techniques to avoid the main problems in the existing unsupervised visual representation frameworks.



# 2. The method and my interpretation

## 2.1. The original method

The paper presents an unsupervised visual representation learning method  with a pretext task of discriminating all images in a dataset using an instance level classifier. The figure below illustrates an overview of the used architecture. ![link text](https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2021/04/Instance_classifier_700px_web.jpg)
This pipeline is similar to common supervised classification, where semantic classes (e.g, car, dog, cat) are replaced with instance IDs. However, the paper presents slight modifications inspired by the recent unsupervised framworks [Chen et al. 2020a](#2), these modifications include stronger data augmentation (e.g, random crop, color jitter, and Gaussian blur), a two-layer MLP pwith non-linearity in between, a cosine softmax loss. where the cosine softmax loss is defined as follows,
<div style="text-align:center"><img src="https://latex.codecogs.com/svg.image?J&space;=&space;-&space;\frac{1}{\vert&space;I&space;\vert}&space;\sum_{i&space;\in&space;I}&space;log&space;\frac{exp(cos(w_i,&space;x_i)/\tau)}{\sum_{j=1}^N&space;exp(cos(w_j&space;x_i)/\tau)}" /></div>

The paper also aims to solve 3 major problems in the vanilla instance classification model for unsupervised representation learning, 1) The large scale isntance classes. 2) The extremely infrequent visiting of instance samples, and 3) the massie number of negative classes that makes the training difficult and slows down the convergence.
Three efficient techniques were propsed to tackle this to improve the representation learning and the scalability:
1. ***Hybrid parallelism.*** To support large scale instance classification, distributed hybrid parallel framework was used to evenly distibute the softmax computations (in both forward and backward passes) to different GPUs as shown in the following figure,
![asdfasdf](https://drive.google.com/uc?export=view&id=1JVcwCJyfZJvrQJqWX_eLp8NO79p_oIX0)

2. ***A contrastive prior.*** A contrastive prior to the instance classifier was introduced to improve the convergence. This is achieved by initializing the classification weights of the as raw instance features extracted by a fixed random network with running BNs.
3. ***Smoothing labels of hardest classes.*** The massive number of negative classes rasises the risk of optimizing over very similar pairs, label smoothing on the top-K hardest instance classes was used to prevent this poblem.

## 2.2. My interpretation 

For my PyTorch implementation, I first implemented a self-supervised contrastive learning model inspired by the [recent frameworks](#2) as a backbone for the implementation. After that, I implemented the sggested components (a contrastive prior, and label smoothing) to solve the problems highlighted by the paper. These two components were implemented as follows,
1. ***For the contrastive prior,*** I trained the randomly initialized network for one epoch while fixing all but BN layers. After that, I assigned the extracted instance features directly to the classification weights.

2. ***For label smoothing.*** Once per epoch, the cosine similarity between the instance the classification weight ![inline1](https://latex.codecogs.com/svg.image?w_i) and all other weights ![inline2](https://latex.codecogs.com/svg.image?w_1,&space;...,&space;w_{i-1},&space;w_{i&plus;1},&space;...,&space;w_N) are comuted, and the top-K similarities are computed. The label of class $j$ is then defined as follows:

<div style="text-align:center"><img src="https://latex.codecogs.com/svg.image?y_j^i&space;=&space;&space;&space;\left\{\begin{array}{ll}&space;&space;&space;&space;&space;&space;1-\alpha&space;&&space;j&space;=&space;i&space;\\&space;&space;&space;&space;&space;&space;\alpha&space;/&space;K&space;&&space;j&space;\in&space;{topK\&space;&space;classes}&space;\\&space;&space;&space;&space;&space;&space;0&space;&&space;otherwise&space;\\\end{array}&space;\right." /></div>



**Note:** For the training process, the DHP traing framework introducted in the paper works for a cluster of GPUs, which is infeasible on my machine. Therefore, this DHP training framework won't be used for training.

# 3. Experiments and results

## 3.1. Experimental setup

In the original paper, the learned representations were evaluated in three ways: First, they fix the representation model and learn a linear classifier upon it (linear evalutation protocol for ImageNet). Second, by evaluating the semi-supervised learning performance, where the methods are required to classify images in the validation set when only a small fraction of data is labeled in the train set. Thrid, by evaluating the transferring performance, this is done by finetutning the representations on several downstream tasks and computing the performance gains.
For this project, I have used the first way (the linear evaluation protocol) to assess the performance on ImageNet. However, for assessment on CIFAR-10, I fitted a downstream classifier using L-BFGS while feeding two augmented views per instance for training (as in the original paper). Accordingly, ResNet-50 was used as the backbone in all the experiments, I trained the model using SGD optimizer, where the weight decay and momentum are set to 0.0001 and 0.9, respectively. The initial learning rate (lr) is set to 0.48 and decays using the cosine annealing scheduler. Morever, I set the temperature in softmax loss to τ = 0.15, and the smoothing factor for labels to α = 0.2 (as provided in the paper).

## 3.2. Running the code

### Running on CIFAR-10

Training
```
$ python3 simclr.py --batch-size 64 --num-epochs 1000 --cosine-anneal --filename output.pth --base-lr 0.48 --test-freq 1
```
evaluation
```
$ python3 lbfgs_linear_clf.py --load-from output.pth
```

### Running on ImageNet
Training
```
$ python3 simclr.py --num-epochs 1000 --cosine-anneal --filename output.pth --test-freq 0 --num-workers 32 --dataset imagenet 
```
evaluation
```
$ python3 gradient_linear_clf.py --load-from output.pth --nesterov --num-workers 32
```

## 3.3. Results

The table below shows the comparison between the obtained results and the results from the original paper:

|         Dataset  | top-1 accuracy (the original paper) | top-1 test accuracy (my implementation) |
|------------------|-------------------------------------|-----------------------------------------|
|         ImageNet | 71.4                               | 68.5                                         |
|         CIFAR-10 | 97.8                              | 95.2                                         | 

# 4. Conclusion

The techniques introduced by the paper were implemented and the results were comparable. However the following points shall be noticed:
1. I haven't used the same parallel training framework and therefore I couldn't reproduce the training on the large-scale datasets used in the paper.
2. Some of the parameters used in the paper (e.g 4096 as a BS), were infeasible as well for the hardware used in this project. Therefore, the results might not be fully comparable to the original paper.

# 5. References


<a id="1">[1]</a> 
Liu, Y., Huang, L., Pan, P., Wang, B., Xu, Y., & Jin, R. (2021). Train a One-Million-Way Instance Classifier for Unsupervised Visual Representation Learning. arXiv preprint arXiv:2102.04848.

<a id="2">[2]</a> 
Chen, T.; Kornblith, S.; Norouzi, M.; and Hinton, G. 2020a.
A Simple Framework for Contrastive Learning of Visual
Representations. arXiv preprint arXiv:2002.05709

# Contact

Please forward your questions and queries to the following e-mail address: [e254278@metu.edu.tr](https://horde.metu.edu.tr/imp/dynamic.php?page=mailbox#:~:text=From%3A-,e254278%40metu.edu.tr,-Date%3A)

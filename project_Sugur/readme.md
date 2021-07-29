# A Casual View on Robustness of Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

This projects aim is to provide an implementation of the paper "A casual view on robustness of neural networks" published in ICLR 2020.

## 1.1. Paper summary

Paper proposes a newly designed deep casual manipulion augmented model called deep CAMA that shows superior robustness against unseen manipulations. More detailed explanation is given in the colab file.

# 2. The method and my interpretation

## 2.1. The original method

Paper tries to create a casual inference between manimulations (M) and inputs in such a way that unseen manipulations are also recognized. The paper introduces three different manipulations and variances (Y, M, Z) on inputs that effects the created outputs and works on intervined manipulations which are represented with M (Further clarifications can be found on Colab file).

## 2.2. My interpretation 

The provided loss function with the usage of ELBO's was too vague. I tried to implement as much as I understood but none of the loss functions performed better than softmax. 

# 3. Experiments and results

## 3.1. Experimental setup

The experiments were tested on two different datasets which were CIFAR 10 and MNIST. For this implementation I only used MNIST since it makes most sense for such manipulations for its inputs. Three different networks are created and trained in the exact same manipoulated data and compared together. All networks are FCN with three layers using relu as activation. First network is trained on non-manipulated data and tested on translated test data. Second network is trained on manipulated data and tested on the same data. Lastly, proposed network is trained and tested on same settings as augmentedly tested network. 

## 3.2. Running the code

Code is implemented and can be run on Colab.

## 3.3. Results

Unfortunately, I could not implement the vagualy explained loss function hence could not compare the proposed network with other FCN's. But all comparisons can be seen from the Colab file. 

# 4. Conclusion

While the general idea of the paper was clear, the abstract and vague explanations, especially on do(m) function was unclear. This made the overall intention of using ELBO and the provided loss function incomprehensible. Also no information for specific input-output pairs or parameters are given.

# 5. References

A casual view on robustness of neural networks
https://openreview.net/forum?id=Hkxvl0EtDH

# Contact

Barış Suğur
sugurbaris@gmail.com

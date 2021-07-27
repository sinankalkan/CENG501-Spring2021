# What Do Neural Networks Learn When Trained With Random Labels?

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper is published at NeurIPS 2020 and its objective is to extend the information about what DNNs (deep neural networks) learn for a given dataset. It examines the outcomes of having networks learn random labels and explains these from a statistical point of view.

## 1.1. Paper summary

The paper first introduces the fact that over-parameterizing DNNs helps them generalize better and makes them be able to learn even a fully random labeled dataset. Thus, there are lots of works that aim to explain differences of training with random and real labels in terms of training time, loss surface, and etc. Also, the authors state that despite these works, what DNNs learn in a random label setting is not thoroughly understood and they give better insights on this topic.

It is showed that training on random labels can have both positive and negative effects which mean faster or slower convergence depending on the training setup and model architecture. Transferring process consists of training on random labels on a subset of the training set (called upstream task) and using the obtained weights on a disjoint subset of the training set with either real or random labels (called downstream task).

It is discussed in the paper that how positive transfer is associated with intrinsic features of the input dataset and when the positive or negative transfer can happen. It is explained that there is a mapping between learned weights of the first layers of the network and dataset and how this mapping can be utilized to understand typical behaviors of DNNs such as which layers solve certain parts of the problem.

### 1.2 Main Contributions

The main contributions of the paper can be summarized as follows:

 - The paper makes experiments for both positive and negative transfer cases for random label training while in previous works, the main focus was the negative effects.
 - The paper interprets what DNNs learn in random label training.
 - In previous works positive transfer coming from bigger weights (because the cross entropy loss is scale-sensitive) was not taken into account. The paper emphasizes this and makes a rescaling before starting the transfer.
 - Alignment phenomenon explains the well-known fact that how earlier layers in DNNs generalize while later layers specialize.
 - The paper proves that what the first layers of a network learn can be approximated by eigenvectors of the input data.
 - Specialization at upper layers may prevent the positive transfer and can be mitigated by increasing model with at those layers.

# 2. The method and my interpretation

## 2.1. The original method

### 2.1.1 Positive transfer

What the paper offers is that the training phase captures the statistical characteristics of a given dataset and it is the reason for positive transfer. To demonstrate this, image patches from the dataset are extracted and the covariance matrix of this data is used to find eigenvectors. A measurement called "misalignment" calculating the deviation between the covariance matrix of data and the covariance matrix of weights learned in the first layers is given in the paper. By using misalignment, authors show that a low misalignment is observed after training both real and random labels whereas there is a high misalignment between weights and some random orthonormal basis. To prove that this result is related to positive effect, Gaussian approximation of weights after random label training is used and positive transfer similar to when random label training weights are directly used occurs. 

Another result derived from the alignment situation is that there is a mapping between eigenvalues of the covariance matrix of image patches and eigenvalues of weights at the first layer. In other words, the first layer learns this mapping function and it is proven in the paper that with different setups and hyperparameters this function's shape is roughly similar. The authors discuss that understanding this function might lead to being able to learn weights by only using input data statistics.

### 2.1.2. Transferring more

Performance of learning transfer can be improved by transferring more layers. In the previous section, only the first layer weights are considered. By treating previous layers' filter weights as the data, more weights can be transferred from the upstream task. Better training accuracy is achieved with this initialization method regardless of the model or dataset used.

### 2.1.3 Negative transfer

Negative transfer in some experiments occurs when the neuron activation values are suddenly and extremely dropped at the point where the training switches to the downstream task. These reductions are observed more in later layers because that earlier layers generalize while the later layers specialize in DNNs. Positive transfer cases, however, are not affected by this problem as much. It is depicted that increasing the number of neurons in the last layers can make the network remain enough capacity for the downstream task and subsequently solve the negative transfer problem.

## 2.2. My interpretation 

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

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

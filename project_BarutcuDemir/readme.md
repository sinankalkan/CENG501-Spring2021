# Beyond Fully Connected Layers with Quaternions: Parametrization of Hypercomplex Multiplications with 1/n Parameters

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

[This paper](https://openreview.net/pdf?id=rcQdycl0zyk) was published as a conference paper at ICLR 2021 by Zhang et al (2021). It was among the 8 Outstanding Papers selected by ICLR organizers out of the 860 papers submitted to the conference.

Although the paper's title is intimidating, the main idea behind it is easy to understand. It builds up on top of the work by Parcollet ([2018](https://hal.archives-ouvertes.fr/hal-02107611/document), [2019](https://core.ac.uk/download/pdf/217859026.pdf)) and Tay ([2019](https://arxiv.org/pdf/1906.04393.pdf)). These papers propose various quaternion networks, which greatly reduce the number of parameters while attaining similar performances with their ordinary counterparts. 

Quaternions are an extension for the vanilla complex numbers, a + bi, which are in the form of, a + bi + cj + dk. One particular area where they are utilized to a great extent is computer graphics, where they can be used to model rotations. In machine learning they are used because the Hamilton product used to multiply two quaternions, serves as a great tool for representation learning. It introduces a certain form of interaction, a bias, between its inputs. 

These interactions are: ijk = i^2 = j^2 = k^2 = -1, ij = k, jk = i, ki = j, ji = -k, kj = -i, ik = -j.  

For a quaternion Q defined as: (the figures belong to Zhang et al. unless otherwise stated)

![Q](https://user-images.githubusercontent.com/62503047/127294554-20573947-34a0-4605-8f46-9904517b3816.PNG)

And a P defined similarly, the Hamilton product of Q and P becomes:

![h](https://user-images.githubusercontent.com/62503047/127294777-8f1312b8-fcb1-4270-bc2a-1a87a972d11d.PNG)

Now, to put this into ML perspective, say there is a weight vector W= R + Xi + Yj + Zk and input X= r + xi + yj + zk. Their Hamilton product is: 

![hh](https://user-images.githubusercontent.com/62503047/127296656-c40eb3d7-9c4d-4055-a052-eb99d34be9d4.PNG)

(Parcollet et al. 2019)

Which can be expressed as:

![Capture](https://user-images.githubusercontent.com/62503047/127296804-faffb983-8093-4bd0-aa3d-9cb43290715e.PNG)

(Parcollet et al. 2019)

Putting aside i,j,k this should seem familiar. It is a weight matrix V times the input X, the standard linear transformation. However the weight matrix V is created using the weight vector W. Normally the weight matrix V, being a 4x4 matrix, would have 16 degrees of freedom however since it is created using the interaction between the input and vector W, which has 4 degrees of freedom. So, quaternion valued linear transformations enjoy a %75 reduction in parameters while achieving similar performance to their real valued counterparts. You can find more details in the aforementioned papers. However the rest of the details such as backprop and loss functions with imarginary numbers are not relevant to this discussion. 

Although they are great in some cases, the problem with quaternion networks is that they can only be used to parametrize certain dimension sizes (4, 8, 16). This hampers their robustness, which brings us to the said paper.

## 1.1. Paper summary

The paper proposes Parametrized Hypercomplex Multiplication Layers to replace fully connected layers. These layers are put together very smartly, such that they subsume both the quaternion layers mentioned previously and the usual linear transformation. Thus the name, Beyond Fully Connected Layers with Quaternions... . The main advantage of the PHM Layer is that they enable choosing an arbitrary n to reduce the number of parameters, whereas this was limited to 4, 8 and 16 with quaternions. Here is how it works:

Fully Connected Layer:

![Screenshot (582)](https://user-images.githubusercontent.com/62503047/127239071-392fe478-671e-483f-8d68-19d791b89b8d.png)

PHM Layer:

![Screenshot (583)](https://user-images.githubusercontent.com/62503047/127239154-8f5d7832-1a26-4b3b-b457-cf5d76b0592d.png)

H: 

![Screenshot (585)](https://user-images.githubusercontent.com/62503047/127239276-69457fd1-c11c-4b4e-997e-2bd22764cedb.png)

The PHM layer resembles a FC layer but differs in the way that the H matrix is obtained. H acts as a weight for input x, but the H matrix is not made up of parameters; the A and S matrices that are used to obtain H are. For a selection of n, which is a hyperparameter, the A matrix takes the form of (n,n,n) whereas the S matrix has the shape (n, dim1/n, dim2/n). The W in a FC layer would be shaped (dim1, dim2) in this case. Kronecker product of every matrix in the tensors A and S are taken and then these are summed to get H. Kronecker product of X and Y is defined as: 

![Screenshot (586)](https://user-images.githubusercontent.com/62503047/127239343-206da384-17b4-4bbc-a6a9-5a35ac76604f.png)

So the process for n=2, dim1=6 and dim2=8 looks like:

![Screenshot (587)](https://user-images.githubusercontent.com/62503047/127239397-3313b5e1-bd4d-4041-951b-42dea44b44d6.png)


The number of parameters here is given by elements in tensor A + tensor B which is n^3 + dim1\*dim2/n. So while 1/n effectively scales the parameters size, there is also a n^3 term which becomes evident with large n. This is further investigated in 2.2. So what does this have to do with quaternions? Remember Figure X. This Hamilton can be expressed as a PHM operation with n=4:

![Screenshot (589)](https://user-images.githubusercontent.com/62503047/127239480-66083189-ca2a-4077-961d-d023ff3b6704.png)

So in theory, the PHM layer can learn the Hamilton product. It can also learn other interactions, so it is more robust than just hardcoding the Hamilton product, which is introducing bias to the system about the interaction between the inputs of the Hamilton product. 

It is important to note that the PHM layer also subsumes feed forward networks when n=1. Tensor A becomes a scalar that scales tensor S, which has the same dimensions with W.
The authors replace some layers of the Transformer and LSTM and conduct several experiments, which are detailed in 2.1.

# 2. The method and our interpretation

## 2.1. The original method

The paper proposes PHM Transformers and LSTM's. They define a PHM Transformer such the following weights are replaced with PHM counterparts:
1. the weights that are multiplied with inputs to produce Query, Key and Value in the beginning of multiheaded attention (called in-projection weights in the official PyTorch implementation)
2. the weights that are multiplied with the concatenation of outputs of the individual heads to reduce their dimension in the end of multiheaded attention (called out-projection weights in the official PyTorch implementation)
3.  the linear layer weights of feedforward network subsequent to the multiheaded attention in encoder/decoder layers

For PHM LSTM:
1. Input gate, Forget gate and cell states are replaced with PHM version. Following linear layers are not changed as stated in paper.
2. Outputs of premise and hypothesis hidden states are concatenated with max and mean pooling heuristic. Resulting vectors are then fed into two layer 300 dimensional multi-layer perceptron. 

## 2.2. Our interpretation 

The paper does a good job of explaining the PHM layer, however misses out on some implementation details. We think this is understandable as this paper sets out to introduce the PHM layer, not the PHM transformer / LSTM. These are just proof of concept networks to show the strength of the PHM layer. They are not achieving SOTA with these networks, so we do not think it is important to match the numbers reported in the paper digit by digit but rather show that the normal transformer and the PHM transformer enjoy close success in certain benchmarks.

There are a bunch of things that are ambiguous about the transformer implementation. These are:
1. **Tokenization**. Nothing is said about the tokenization method. There are many tokenization methods to choose from today because they can really improve the performance of the network. We chose to use the most elementary method that is word tokenization. Every unique word is assigned a token along with some special tokens such as beginning of sentences, end of sentence and unknown word token. It is important to note that virtually no language model uses this tokenization method today, they opt in for subword tokenization methods such as byte pair encoding and Sentencepiece. These are easy to use and we have experience working with them but we chose word tokenization due to its simplicity and the ambiguity of the tokenization method in the paper.
2. **Experimental Setup**. Nothing is said about the chosen optimizer and its parameters. The batch size is also reported for only natural language inference (LSTM). There is no explanation about hidden sizes of both LSTM and MLP, and choice of activation function between linear layers in NLI tasks. The authors report how much they train the networks in terms of step sizes, but since the batch size is ambiguous for most tasks, this does not help to replicate the results. Our choice of these hyperparameters are explained in the experimental setup.
3. **Model Architecture**. Some choices regarding the model architecture are ambiguous such as the feed forward network hidden size. Our choice of these hyperparameters are explained in the experimental setup.
4.  **Concatenation Process in LSTM**. There is no explanation about concatenation of *premise* and *hypothesis* outputs in NLI experiments. We inferred from previous works for concatenation process.
5.  **BLEU smoothing function**. The BLEU metric used for translation is not a perfect metric and some modifications have been proposed to make it closer to human judgement. One such proposal is the use of [smoothing functions](https://leimao.github.io/blog/BLEU-Score/). We used a particular but common smoothing function, but the authors have not made it clear whether they have used it or which one. So a comparison between our BLEU scores and the authors' may not be %100 accurate.
6.   Concatenation heuristic method [Qian Chen et al., 2017.](https://arxiv.org/abs/1609.06038) is used in previous work. We used this heuristic for our implementation in NLI task. Concatenation process is done by computing both average and max pooling, and concatenate all these vectors to form the final fixed
length vector v. The final fixed length vector v is calculated as follows:

  ![Concat](https://user-images.githubusercontent.com/84293711/127313944-0d5a2d55-a0e2-4f96-a74f-d92f24ce826c.jpg)

  (Qian Chen et al., 2017)

Where *Va,ave* and *Va,max* are LSTM outputs of premise sentences applied by average   and max operations respectively along hidden size direction, and *Vb,ave* and *Vb,max* are the outputs of hypothesis outputs of same process. 


*There is one particular aspect of the paper that is misleading.* The paper makes the overall architecture of the PHM transformer very clear. They report a reduction parameters as you can see in the tables at 3.3 Results. However it is impossible to reduce the total number of parameters of a transformer by 1/n by only replacing the parts mentioned in 2.1 because there are also word embeddings that map the tokens to the encoder and the generator that maps the decoder to the tokens. In fact these may make up the bulk of a transformer depending on the vocabulary size and the model setup. **So we propose a PHM+ transformer that also replaces these layers with PHM weights.** 

In PHM+ transformer every weight is replaced by its PHM counterpart, therefore the model is truly scalable by 1/n unlike the PHM transformer in the paper. We carried out our work using PHM+ transformers, including the results reported in the following section. We did not feel the need to use the authors' version as the PHM+ transformer sufficed to replicate their results with less parameters.    

# 3. Experiments and results

For transformer experiments we chose to reproduce the style transfer, because the data was the easiest to obtain, and the De-En translation task, because it was the one the paper went into farthest detail. In style transfer, the task is to translate modern English to Shakespearean English whereas in De-En it is to translate German to English.

## 3.1. Experimental setup

Notice how every weight dimension hyperparameter is an exponential of 2, so that we can pick n=2,4,8...256.

### For style transfer:

Setup Component | The authors | Us
------------ | ------------- | -------------
Optimizer | - | AdamW
Learning Rate | - | 0.0001 
Weight Decay | - | 0.05
Tokenization | - | word level
Vocabulary Size | - | 8192
Batch Size | - | 32
Training Steps | 10,000 | 5,175*
Embedding Size | 512** | 512
Feed Forward Hidden Size | 512** | 512
Number of Heads | 8 | 8
Number of Encoder Layers | 4 | 4
Number of Decoder Layers | 4 | 4
Beam Size | 5 | 5
Beam Search Length Penalty Exponent (alpha) | 0.6 | 0.6

The data is available [here](https://github.com/tlatkowski/st).

The loss curves can be accessed at our [Weights and Biases report for style transfer](https://wandb.ai/demegire/transformer/reports/Style-Transfer--Vmlldzo4ODgzMTQ).

![Training Curves](https://user-images.githubusercontent.com/62503047/127185010-28cb14ca-7b6e-47e3-8c85-29abef553a03.png)

![Validation Curves](https://user-images.githubusercontent.com/62503047/127185236-5c748f9c-98ea-4acf-bfdd-908f985aca9f.png)

The step numbers here are not the real steps, rather they are the intervals for reporting the losses to Weights and Biases.

### For De-En translation:

Setup Component | The authors | Us
------------ | ------------- | -------------
Optimizer | - | AdamW
Learning Rate | - | 0.0001 
Weight Decay | - | 0.05
Tokenization | - | word level
Vocabulary Size | - | 32768
Batch Size | - | 16
Training Steps | 50,000 | 50,000***
Embedding Size | 256** | 256
Feed Forward Hidden Size | 256** | 256
Number of Heads | 4 | 4
Number of Encoder Layers | 2 | 2
Number of Decoder Layers | 2 | 2
Beam Size | 5 | 5
Beam Search Length Penalty Exponent (alpha) | 0.6 | 0.6

The data is available [here](http://www.statmt.org/wmt14/translation-task.html). We only use the Europarl v7 dataset whereas the authors also used Common Crawl Corpus and the News Commentary.

The loss curves can be accessed at our [Weights and Biases report for translation](https://wandb.ai/demegire/transformer/reports/Translation--Vmlldzo4ODgzMTE).

![W B Chart 7_28_2021, 1 38 15 PM](https://user-images.githubusercontent.com/62503047/127309563-87e06b18-e832-4b16-bee3-13009c5f42c7.png)

![W B Chart 7_28_2021, 1 38 26 PM](https://user-images.githubusercontent.com/62503047/127309583-4b0ba188-4f7d-43d9-b18a-669bdcf51a8f.png)


\* We do early stopping at 5,175 because the model starts to overfit. The training curves for the models can be viewed above. 

\** The authors report a 'hidden size' for the transformer and we used this value both for feed forward network hidden size and the embedding size.

\*** At 50,000 steps we have barely covered the Europarl dataset, which is 1/3 of the authors' dataset. So the model in the paper is presumably trained for much longer although we can't be sure because the batch size is not reported along the training steps. There is a batch size given in the inference task for the LSTM, 256, which is much bigger than our 16. 


### For NLI task on MNLI Dataset
Setup Component | The authors | Us
------------ | ------------- | -------------
Optimizer    | Adam | Adam
Weight Decay | - | -|
Batch Size   | 256 | 256 |
Epochs       | - | 10 (I)|
Max Lenght of Input | - | 128 |
Feed Forward Hidden Size | - | 100 |
LSTM Hidden Size | 300 | 300 |
n            | 2,5,10 | 10 |

(I): We defined the number of epochs as 10 for training process but we were not able to finish the train due to limited runtime on Colab.

## 3.2. Running the code

The Colab notebook for style transfer thoroughly explains the parts of the code. It is very straightforward since it is on Colab. Translation is also available but since it shares much of its codebase with style transfer, it is not explained in detail.

**Style Transfer** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bM_Gqw3V6q-gEu7KkzxVEYI3qimlKHPe?usp=sharing)

**Translation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x64OkSUJdJgx-2FN95KtCZ7QRQfDM8Su?usp=sharing)

## 3.3. Results

### For style transfer:

The results reported by the authors:

Model | Parameters | BLEU
------------ | ------------- | -------------
Transformer | 44M | 11.65
PHM-Transformer (n=2) | 22M | 12.20 
PHM-Transformer (n=4) | 11M | **12.42**
PHM-Transformer (n=8) | 5.5M | 11.66
PHM-Transformer (n=16) | 2.9M | 10.76 

Our results:

Model | Parameters | BLEU | Training time for 1 epoch (sec) | Inference time for 500 lines (sec)
------------ | ------------- | ------------- | ----------- | -------------
Transformer | 29,421,568 | 17.52 | 41.644 | 205
PHM-Transformer (n=2) | 13,857,048 | 16.55 | 50.492 | 344
PHM-Transformer (n=4) | 7,043,264 | 15.39 | 55.223 | 327
PHM-Transformer (n=8) | 3,651,072 | 16.09 | 63.436 | 358
PHM-Transformer (n=16) | 2,072,576 | 16.36 | 81.249 | 595

### For De-En translation:

The results reported by the authors:

Model | Parameters | BLEU | Train time per 100 steps (sec) | Inference time (sec)
------------ | ------------- | ------------- | ------------- | -------------
Transformer | 44M | **36.68** | 7.61 | 336
PHM-Transformer (n=2) | 22M | 35.52 | - | -
PHM-Transformer (n=4) | 11M | 35.53 | 7.92 | 299
PHM-Transformer (n=8) | 5.5M | 34.16 | 7.70 | 282
PHM-Transformer (n=16) | 2.9M | 33.89 | - | -

Our results:

Model | Parameters | BLEU | Train time for 50,000 steps (sec) | Inference time for 500 lines (sec)
------------ | ------------- | ------------- | ------------- | ------------- 
Transformer | 27,309,056 | 17.91 | 2257 | 257
PHM-Transformer (n=2) | 13,579,416 | 17.89 | 2604 | 392
PHM-Transformer (n=4) | 6,830,272 | 17.30 | 2745 | 438 
PHM-Transformer (n=8) | 3,463,680 | 18.02 | - | 553 
PHM-Transformer (n=16) | 1,844,224 | 17.65 | 5003 | 980

### For NLI Tasks

We were not able to get the results from trained models because of runtime limitation and shortage of memory in Colab. However, our implementation can be completed with sufficient amount of resources. We showed that training loss decreases through iterations when PHM-LSTM implementation is used.

# 4. Conclusion

**We were able to reproduce the authors' results in style transfer and De En translation tasks. Although we could not match their exact numbers because of training limitations and hyperparameter ambiguities, we saw that the vanilla transformer and the PHM+ transformer yielded very similar performances, which was the point of the paper.**

Interestingly we outperformed the authors' model in style transfer. We think this is due to the choice of BLEU smoothing function mentioned before but we would also like to note that the authors report a large number of steps for training for this task. In fact we overfit the data with the authors' number of steps with a small batch size (see the figure under style transfer experiment), so when this large number of steps is coupled with a large batch size it could lead to greater overfitting. But we think this is extremely unlikely given the authors' experience, so there is probably something else at play.

We observed a lower performance than the authors' in the translation task. This was expected as we trained the same number of steps, with presumably a much smaller batch size. In fact we could not cover 1/3 of the dataset and there was still room for improvement. 

One final thing of interest is the training and inference times for both tasks. When n is increased, the number of parameters decrease while the number of operations increase. One of these, may dominate the other. We observed that both times went up when n was increased while the authors reported a decrease. This hints that the authors' have a more efficient PHM layer code.

Due to limited memory and unparalizable structure of LSTM network, we only managed to train PHM-LSTM network half of a epoch. Even with a half epoch, we showed that PHM network is learning after a few iterations. Training process can be speed up by converting words into embeddings before training. Also, code of PHM layers can further be optimized. 

To wrap up, we enjoyed reproducing this paper and learned a lot during the process. We would like to thank the authors for writing such a great and mostly clear paper and Sinan Hoca for equipping us with the skills to take on this project.

# 5. References

Adina Williams, Nikita Nangia, Samuel R. Bowman. 2017. *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference*. https://arxiv.org/abs/1704.05426

Aston Zhang, Yi Tay, Shuai Zhang, Alvin Chan, Anh Tuan Luu, Siu Cheung Hui, Jie Fu. 2021. *Beyond Fully-Connected Layers With Quarternions: Parameterization Of Hypercomplex Multiplication With 1/n Parameters*. https://arxiv.org/abs/2102.08597

Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning. 2015. *A large annotated corpus for learning natural language inference*. https://arxiv.org/abs/1508.05326

Sepp Hochreiter, Jurgen Schmidhuber. Long short-term memory. ¨ Neural computation, 9(8):
1735–1780, 1997.

Titouan Parcollet, Mohamed Morchid, and Georges Linares. Quaternion convolutional neural net-
works for heterogeneous image processing. In ICASSP 2019-2019 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), pp. 8514–8518. IEEE, 2019

Titouan Parcollet, Ying Zhang, Mohamed Morchid, Chiheb Trabelsi, Georges Linares, Renato
De Mori, and Yoshua Bengio. Quaternion convolutional neural networks for end-to-end auto-
matic speech recognition. arXiv preprint arXiv:1806.07789, 2018.

Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. 2017. *Enhanced LSTM for Natural Language Inference*. https://arxiv.org/abs/1609.06038

Yi Tay, Aston Zhang, Luu Anh Tuan, Jinfeng Rao, Shuai Zhang, Shuohang Wang, Jie Fu, and
Siu Cheung Hui. Lightweight and efficient neural natural language processing with quaternion
networks. arXiv preprint arXiv:1906.04393, 2019

# Contact

Ege Demir - demegire@gmail.com
Mehmet Barutcu - mehmetbarutcu00@gmail.com 

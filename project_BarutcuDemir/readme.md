# Beyond Fully Connected Layers with Quaternions: Parametrization of Hypercomplex Multiplications with 1/n Parameters

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

[This paper](https://openreview.net/pdf?id=rcQdycl0zyk) was published as a conference paper at ICLR 2021 by Zhang et al. It was among the 8 Outstanding Papers selected by ICLR organizers out of the 860 papers submitted to the conference.

Although the paper's title is intimidating, the main idea behind it is easy to understand. It builds up on top of the work by Parcollet ([2018](https://hal.archives-ouvertes.fr/hal-02107611/document), [2019](https://core.ac.uk/download/pdf/217859026.pdf)) and Tay ([2019](https://arxiv.org/pdf/1906.04393.pdf)). These papers propose various quaternion networks, which greatly reduce the number of parameters while attaining similar performances with their ordinary counterparts. For context, quanternions are an extension for the vanilla complex numbers, a + bi, which are in the form of, a + bi + cj + dk. One particular area where they are utilized to a great extent is computer graphics, where they can be used to model rotations. In machine learning they are used because the Hamilton product used to multiply two quaternions, serves as a great tool for represantation learning. Hamilton product is defined as ... But the problem with quaternion networks is that they can only be used to parametrize certain dimension sizes (4, 8, 12). This limits their potential, which brings us to the said paper.

## 1.1. Paper summary

Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and our interpretation

## 2.1. The original method

The paper proposes PHM Transformers and LSTM's. They define a PHM Transformer such the following weights are replaced with PHM counterparts:
1. the weights that are multiplied with inputs to produce Query, Key and Value in the beginning of multiheaded attention (called in-projection weights in the official PyTorch implementation)
2. the weights that are multiplied with the concatenation of outputs of the individual heads to reduce their dimension in the end of multiheaded attention (called out-projection weights in the official PyTorch implementation)
3.  the linear layer weights of feedforward network subsequent to the multiheaded attention in encoder/decoder layers


## 2.2. Our interpretation 

The paper does a good job of explaining the PHM layer, however misses out on some implementation details. We think this is understandable as this paper sets out to introduce the PHM layer, not the PHM transformer / LSTM. These are just proof of concept networks to show the strength of the PHM layer. They are not achieving SOTA with these networks, so we do not think it is important to match the numbers reported in the paper digit by digit but rather show that the normal transformer and the PHM transformer enjoy close success in certain benchmarks.

There are a bunch of things that are ambiguous about the transformer implementation. These are:
1. **Tokenization**. Nothing is said about the tokenization method. There are many tokenization methods to choose from today because they can really improve the performance of the network. We chose to use the most elementary method that is word tokenization. Every unique word is assigned a token along with some special tokens such as beginning of sentences, end of sentence and unknown word token. It is important to note that virtually no language model uses this tokenization method today, they opt in for subword tokenization methods such as byte pair encoding and Sentencepiece. These are easy to use and we have experience working with them but we chose word tokenization due to its simplicity and the ambiguity of the tokenization method in the paper.
2. **Experimental Setup**. Nothing is said about the chosen optimizer and its parameters. The batch size is also reported for only natural language inference (LSTM). The authors report how much they train the networks in terms of step sizes, but since the batch size is ambiguous for most tasks, this does not help to replicate the results. Our choice of these hyperparameters are explained in the experimental setup.
3. **Model Architecture**. Some parameters regarding the model architecture are missing such as the feed forward network hidden size. Our choice of these hyperparameters are explained in the experimental setup.

*There is one particular aspect of the paper that is misleading.* The paper makes the overall architecture of the PHM transformer very clear. They report a reduction parameters as you can see in the tables at 3.3 Results. However it is impossible to reduce the total number of parameters of a transformer by 1/n by only replacing the parts mentioned in 2.1 because there are also word embeddings that map the tokens to the encoder and the generator that maps the decoder to the tokens. In fact these may make up the bulk of a transformer depending on the vocabulary size and the model setup. **So we propose a PHM+ transformer that also replaces these layers with PHM weights.** In this version every weight is replaced by its PHM counterpart.

# 3. Experiments and results

## 3.1. Experimental setup

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

\* We do early stopping at 5,175 because the model starts to overfit. The training curves for the models can be viewed above. 
\** The authors report a 'hidden size' for the transformer and we used this value both for feed forward network hidden size and the embedding size.
\*** At 50,000 steps we have barely covered the Europarl dataset, which is 1/3 of the authors' dataset. So the model in the paper is presumably trained for much longer although we can't be sure because the batch size is not reported along the training steps. There is a batch size given in the inference task for the LSTM, 256, which is much bigger than our 16. 

## 3.2. Running the code

The Colab notebooks for style transfer and translation thoroughly explains the parts of the code. It is very straightforward since it is on Colab.

Style Transfer [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bM_Gqw3V6q-gEu7KkzxVEYI3qimlKHPe?usp=sharing)]

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
------------ | ------------- | ------------- | ------------- | -------------
Transformer | 29,421,568 | 17.86 | x | x
PHM-Transformer (n=2) | 13,857,048 | 16.55 | 50.492 | x
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

Model | Parameters | BLEU | Inference time for 500 lines (sec)
------------ | ------------- | ------------- | ------------- 
Transformer | 27,309,056 | 16.62 | x 
PHM-Transformer (n=2) | 13,579,416 | 17.34 | x
PHM-Transformer (n=4) | 6,830,272 | 17.30 | x 
PHM-Transformer (n=8) | 3,463,680 | 18.02 | 553 
PHM-Transformer (n=16) | 1,844,224 | x | x 

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Ege Demir - demegire@gmail.com
Mehmet Barutçu - 

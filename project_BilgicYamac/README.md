# Joint Learning of Answer Selection and Answer Summary  Generation in Community Question Answering

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

## 1.1. Paper summary

There have been many studies on different tasks in Community question answering (CQA) such as answer selection, question-question relatedness and comment classification. The paper focuses on answer selection and the challenges that CQA brings which are the noise introduced by the redundancy and the lengthiness of answers in CQA. The contributions of the paper can be summarized as follows:
1. A novel joint learning framework of answer selection and abstractive summarization is proposed.
2. A new dataset, WikiHowQA, is proposed which can be adapted to answer selection and summarization.
3. For the above-mentioned tasks, we provide a transfer learning strategy without reference answer summaries.

# 2. The method and my interpretation

## 2.1. The original method

The aim is to jointly conduct answer selection and abstractive summarization. The overall framework of Answer Selection and Abstractive Summarization consists of four components: shared Compare-Aggregate Bi-LSTM Encoder, sequence-to-sequence Model with Question-aware Attention, question Answer Alignment with Summary Representations, question-driven Pointer-generator Network.

Shared Compare-Aggregate Bi-LSTM Encoder: The word embeddings of the question and the original answer are preprocessed to obtain a new embedding vector which also captures some contextual information in addition to the word itself. This new word embeddings are provided into the Bi-LSTM encoder. 

![image](https://user-images.githubusercontent.com/57533312/127110893-8dec128e-9004-48c9-89ba-47083a7254e8.png)

Seq2Seq Model With Question-aware Attention: A question-aware attention based seq2seq model to decode the encoded sentence representation of the answer is proposed. A unidirectional LSTM is adopted as the decoder. Moreover, an attention mechanism is proposed.

Question Answer Alignment with Summary Representations: A two-way attention mechanism is proposed to generate the co-attention between the encoded question representation and the decoded summary representation. With a learnable attention parameter matrix U, the attention matrices are obtained. The dot product of the attention vectors and question/summary representations give the final attentive sentences.

Question-driven Pointer-generator Network: First, the probability distribution Pvocab over the fixed vocabulary is obtained by passing the summary representation through a softmax layer.

![image](https://user-images.githubusercontent.com/57533312/127110973-d336f7bb-2557-4b9f-baf5-bd454dea9559.png)

Then a question-aware pointer network is proposed to copy words from the source with guidance of the question information. It uses information of the decoded summary representation, decoder input and question representation.
A pointer generator network allows to obtain the final probability distribution over the fixed vocabulary and the words from the source article.

![image](https://user-images.githubusercontent.com/57533312/127111031-ccfd1aac-eae9-499a-b87c-45bfab987ec8.png)

## 2.2. My interpretation 

Our design for the solution includes preprocessing of word embeddings, Shared Compare-Aggregate Bi-LSTM Encoder, and Seq2Seq Model with Question-aware Attention. However, the given WikiHowQA dataset doesn't conclude original answer/summary pairs. Therefore, the modules related to summarization could not be added.  In the original paper, the last part of the Seq2Seq Model with Question-aware Attention were not clearly defined to generate summary representation. Since, Question Answer Alignment with Summary Representations requires decoded summary representations, this part could not be reproduced also.

**Preprocessing of word embeddings**: 

Below equations give the result for given module.

![image](https://user-images.githubusercontent.com/57533312/127489930-57c347fc-822f-4405-856b-ed1e76ec1cd3.png)

**Bi-LSTM encoders**:
	
These are the encoders to get the contextual information with question and answer word embeddings.

**Seq2Seq Model with Question-aware Attention**:
	
An LSTM decoder and several matrix operations took place to get an attention matrix according to the question and answer embeddings. Related equations were given below:

![image](https://user-images.githubusercontent.com/57533312/127490011-4c0b67f1-21fe-4580-8067-b7d063a2e2c3.png)	

&nbsp; &nbsp; &nbsp;![image](https://user-images.githubusercontent.com/57533312/127490053-510e5b3b-b66d-410c-bc3f-9c5060759c18.png)


# 3. Experiments and results

## 3.1. Experimental setup

All implemented models are trained with pre-trained GloVE embeddings of 100 dimensions with a vocabulary size of 50k. During training and testing, the length of the articles are limited to 400 words while summaries are limited to 100. Based on the answer selection evaluation result on the validation set, early stopping is applied. Implementation details of the original paper state that the answer selection model is trained for 5 epochs while summarization model is trained for 20 epochs. The model is trained with a learning rate 0.15 with initial accumulator value of 0.1 and batch size of 32. The dropout rate is set to 0.5. Hidden unit sizes of Bi-LSTM encoder and the LSTM decoder are set to 150.

## 3.2. Running the code

The project structure is given below:
```
project_Acir
???????????? code/
???    ???????????? main.py
???    ???????????? preprocessing.py
???    ???????????? seq2seq.py
???    ???????????? data
|         ???????????? sequence
|    ???????????? WikiHowQACorpus
|    |    ???????????? summary.txt
|    |    ???????????? test.txt
|    |    ???????????? train.txt
|    |    ???????????? valid.txt
???    ???????????? glove
???  README.md
```

- WikiHowQACorpus dataset and GloVE should be downloaded from:
      https://github.com/dengyang17/wikihowQA
      https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
- If data/sequence folder is not in your repository. Create empty sequence folder under data/ directory.
- main.py should be run to start training.

## 3.3. Results

Evaluation 
The proposed method compared four aspects: "Answer Selection," "Analysis of the Length of Answer", "Answer Summary Generation" and "Human Evaluation on Summarization".

For the "Answer Selection" part, MRR and MAP evaluation metrics are used. In this evaluation, Siamese BiLSTM (Mueller and Thyagarajan 2016), AttBiLSTM (Tan et al. 2016), AP-LSTM (dos Santos et al. 2016), CA (Compare-Aggregate) (Wang and Jiang 2017), and COALA (Ruckl ?? e, Moosavi, and Gurevych 2019) methods are compared with the proposed method. The result of this comparison is shown at below, and ASAS, which is the proposed method, gives a better result.

![Table1](https://user-images.githubusercontent.com/45417780/127491683-ff8b2336-a076-4b2d-90d8-a513d192c2ee.PNG "Title")

In terms of text summarization, the proposed method (ASAS) also gives better results in tables shown below.
![Table2](https://user-images.githubusercontent.com/45417780/127491730-6245d515-b9a4-472b-86f5-7416e7a58965.PNG "Title")

![Table3](https://user-images.githubusercontent.com/45417780/127491752-ab99131a-88dc-42e7-8142-7ba71cebe7ea.PNG "Title")

The final evaluation is about the length and accuracy. At below, we can see the accuracy with respect to answer length. This accuracy is compared with AP-LSTM and Compare-Aggregate Model. ASAS also gives better results in this perspective.

![Figure1](https://user-images.githubusercontent.com/45417780/127491789-ef43b43e-a027-4fe8-8486-027534c8125d.PNG "Title")

# 4. Conclusion

In this project, the paper *Joint Learning of Answer Selection and Answer Summary Generation in Community Question Answering* [1] is tried to be implemented. The original paper has the purpose of answer selection for community questions with abstractive summarization of answers. 

Since there was a lack of data (answer-summary pairs were not clearly provided) one of the two main purposes of the paper could not be reproduced. Besides, the methods and models were customized and the implementation details were not clearly given. Therefore, in our embedding, we were just sticked to the equations and tried to implement same matrix operations. 

# 5. References

[1]: [Deng, Yang, Wai Lam, Yuexiang Xie, Daoyuan Chen, Yaliang Li, Min Yang and Ying Shen. ???Joint Learning of Answer Selection and Answer Summary Generation in Community Question Answering.??? AAAI (2020).](https://arxiv.org/pdf/1911.09801)

[2]: [Wang, S., & Jiang, J. (2017). A Compare-Aggregate Model for Matching Text Sequences. ArXiv, abs/1611.01747.](https://arxiv.org/pdf/1611.01747)

# Contact

Abdullah ??mer Yama?? - omer.yamac@metu.edu.tr

Alper Bilgi?? - alper.bilgic@metu.edu.tr

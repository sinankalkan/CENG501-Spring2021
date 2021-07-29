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

Below equation is the result for given module. 
![image](https://user-images.githubusercontent.com/57533312/127489930-57c347fc-822f-4405-856b-ed1e76ec1cd3.png)

**Bi-LSTM encoders**:
	
These are the encoders to get the contextual information with question and answer word embeddings.

**Seq2Seq Model with Question-aware Attention**:
	
An LSTM decoder and several matrix operations took place to get an attention matrix according to the question and answer embeddings. Related equations were given below
![image](https://user-images.githubusercontent.com/57533312/127490011-4c0b67f1-21fe-4580-8067-b7d063a2e2c3.png)	
![image](https://user-images.githubusercontent.com/57533312/127490053-510e5b3b-b66d-410c-bc3f-9c5060759c18.png)


# 3. Experiments and results

## 3.1. Experimental setup

All implemented models are trained with pre-trained GloVE embeddings of 100 dimensions with a vocabulary size of 50k. During training and testing, the length of the articles are limited to 400 words while summaries are limited to 100. Based on the answer selection evaluation result on the validation set, early stopping is applied. Implementation details of the original paper state that the answer selection model is trained for 5 epochs while summarization model is trained for 20 epochs. The model is trained with a learning rate 0.15 with initial accumulator value of 0.1 and batch size of 32. The dropout rate is set to 0.5. Hidden unit sizes of Bi-LSTM encoder and the LSTM decoder are set to 150.

## 3.2. Running the code

The project structure is given below:
```
project_Acir
│─── code/
│    │─── main.py
│    │─── preprocessing.py
│    │─── seq2seq.py
│    │─── data
|         │─── sequence
|    │─── WikiHowQACorpus
|    |    │─── summary.txt
|    |    │─── test.txt
|    |    │─── train.txt
|    |    │─── valid.txt
│    │─── glove
│  README.md
```

- WikiHowQACorpus dataset and GloVE should be downloaded from:
      https://github.com/dengyang17/wikihowQA
      https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
- If data/sequence folder is not in your repository. Create empty sequence folder under data/ directory.
- main.py should be run to start training.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Provide your names & email addresses and any other info with which people can contact you.

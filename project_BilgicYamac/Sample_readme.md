# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper Joint Learning of Answer Selection and Answer Summary Generation in Community Question Answering was presented in AAAI 2020. The purpose of the algorithm is to pick the best answer for a "How to" question and provide the summary of that answer. As the paper also focus on, community question answering (CQA) attract more attention of both academy and industry lately. However, there are several challenges that still limit the performance. These issues may be counted as, the redundancy and lengthiness of answers. While these issues are limiting the performance, they are also keeping answers unclear for community users. The paper aims to achive is to tackle the answer selection performance issues while generating a summary of answers.

## 1.1. Paper summary

There are many studies have been made on different tasks in Community question answering (CQA) such as answer selection, question-question relatedness and comment classification. The paper focuses on answer selection and the challenges that CQA brings which are the noise instroduced by the redundancy and the lengthiness of answers in CQA. The contributions of the paper can be summarized as follows:
1. A novel joint learning framework of answer selection and abstractive summarization is proposed.
2. A new dataset, WikiHowQA, is proposed which can be adapted to answer selection and summarization.
3. A transfer learning strategy which enables mentioned tasks without reference answer summaries to conduct is provided.

# 2. The method and my interpretation

## 2.1. The original method

The aim is jointly conduct answer selection and abstractive summarization. The overall framework of Answer Selection and Abstractive Summarization consists of four components. Shared Compare-Aggregate Bi-LSTM Encoder, Sequence-to-sequence Model with Question-aware Attention, Question Answer Alignment with Summary Representations, Question-driven Pointer-generator Network.
**Shared Compare-Aggregate Bi-LSTM Encoder**: The word embeddings of question and the original answer are preprocessed to obtain a new embedding vector which also captures some contextual information in addition to the word itself. The this new word embeddings are provided into the Bi-LSTM encoder. 

![image](https://user-images.githubusercontent.com/57533312/127110893-8dec128e-9004-48c9-89ba-47083a7254e8.png)

**Seq2Seq Model With Question-aware Attention**: A question-aware attention based seq2seq model to decode the encoded sentence representation of the answer is proposed. A unidirection LSTM is adopted as the decoder. Also an attention mechanism is proposed.

**Question Answer Alignment with Summary Representations**: A two way attention mechanism is proposed to generate the co-attention between the encoded question representation and the decoded summary representation. With a learnable attention parameter matrix U, the attention matrixes are obtained. The attention vectors and question/summary represantations are dot producted to get final attentive sentences.

**Question-driven Pointer-generator Network**: First, the probability distribution Pvocab over the fixed vocabulary is obtained by passing the summary representation through a softmax layer.

![image](https://user-images.githubusercontent.com/57533312/127110973-d336f7bb-2557-4b9f-baf5-bd454dea9559.png)

Then a question-aware pointer network is proposed to copy words from the source with guidance of the question information. It uses information of decoded summary representation, decoder input and question representation.
And a pointer generator network allows to obtain the final probability distribution over the fixed vocabulary and the words from the source article.

![image](https://user-images.githubusercontent.com/57533312/127111031-ccfd1aac-eae9-499a-b87c-45bfab987ec8.png)

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

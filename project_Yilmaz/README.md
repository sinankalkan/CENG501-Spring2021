# Continuous Representations of Intents for Dialogue Systems

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper "Continuous Representations of Intents for Dialogue Systems" \[1\] addresses the problem of representing user intents in conversational assistants. The main idea is to map every intent class to a point in space that is available through a set of coordinates and base matrices. In case of an encounter with an unseen intent, the model is able to represent it within the calculated space and classify it in the zero-shot learning setup.

## 1.1. Paper summary

The paper introduces the notion of intent space and proposes a new zero-shot detection algorithm. It uses recurrent neural network structures and an open-source [dataset](https://github.com/sonos/nlu-benchmark). The evaluation metric is accuracy and the results are reported after a single fold evaluation.

# 2. The method and my interpretation

## 2.1. The original method

The original method is described in rather generic terms where an intent space with a set of bases and coordinates for each intent is able to represent every intent distinctly. In order for this parameter sets, bases and coordinates, to be learnable, the paper proposes to use an RNN structure and encode the base information within the hidden state multiplier of the RNN. Namely, if h is the hidden state of the RNN and U * h is the feed forward calculation for the RNN, U is the base matrix chosen for the related class. Bases of the space are taken as matrices to increase modeling power with use of more parameters.

## 2.2. My interpretation 

Based on the explanations in the paper, I assumed that there exist several different RNN modules for each intent class available during training since the base for a given intent needs to be specific to that intent. The paper also states that there exist different initial hidden vectors for each intent class, therefore, I assumed every RNN module uses a different initial hidden state vector (randomly initialized).

I assumed that the output of each RNN module is a base vector (I did not implement base matrices for the sake of simplicity) whose size is equal to the number of seen classes during training. This size constraint is also mentioned in the paper. In order to achieve this, a fully connected layer maps the 128 dimensional output of the RNN cell to the number of seen intents (6 for the SNIPS dataset).

The obtained base matrices for a given input is multiplied by the coordinates matrix which is initialized as the identity matrix. During training, as explained in the paper, the base matrices are learned for the first 5 epochs while the coordinates are frozen, after which base matrices are frozen and coordinates are learned for another 5 epochs. Lastly, base matrices are updated again while coordinates are frozen for 5 epochs. This training procedure uses only seen intents during training.

After 15 epochs, the regularization term calculated over a small set of unseen intents is used to update the parameters of what is described as the expansion matrices in the paper. These matrices are of the same size with the base matrices and allow higher modeling power with introduction of extra parameters. During this second training phase, which lasts for another 15 epochs, bases and coordinates are frozen and only the expansion matrices are updated. I implemented an expansion matrix for every base although expansion matrices are defined only for unseen intents in the paper. However, I ended up using only the expansion matrix corresponding to the unseen intent similar to the description in the paper.

The regularization term on the unseen intents described in the paper fails to train the model in my experiments, so I employed Cross Entropy Loss for the set of unseen intents. I used all available unseen intent data in the training set.

# 3. Experiments and results

## 3.1. Experimental setup

I used the pretrained 300 dimensional GloVe embeddings \[2\] over 840B tokens, but filtered out all the words that do not appear in the dataset and saved the remaining word vectors under `used_vectors.pkl` file in order to reduce the size. Unknown words are mapped to the average word vector of the entire GloVe corpus.

SGD optimizer with learning rate 0.1 is used for the experiments with Cross Entropy Loss. I observed Adam to be failing, possibly due to the weight decay introduced on the coordinate parameters. At each epoch, early stopping is employed when 0.85 accuracy is obtained on the validation set, calculated every 1000 steps. The batch size is chosen as 1 for the experiments in order to simplify calculations of matrix multiplications. Weight decay on the coordinate parameters is chosen as 1e-5. The unseen intent is chosen as "GetWeather" as in the paper for the algorithm in `model.py`.

During training, I observe a severe drop in validation accuracy at the start of 11th epoch, which happens after the bases are learned for 5 epochs and coordinates are learned in the following 5 epochs. For this 10-epoch period, validation accuracy increases monotonically but at the start of the 11th epoch, there is a dramatical drop. This is not reported in the paper since the reported accuracy values are those at the end of the epochs. The validation accuracy climbs back to its initial value by the end of the epoch in my experiments as well. 

Based on this observation, I abandoned the 15-epoch training schedule explained in the text of the paper and adopted the training schedule described in Figure 3 of the paper. The schedule is learning bases for the first 3 epochs, then learning coordinates for 2 epochs, then learning bases for 5 epochs and then learning only the expansion matrices for the rest of 30 epochs. After the epoch 7, unseen intents are used to calculate the regularization term and added to the loss term. The multiplier on the regularizer loss is 0.2 as stated in the paper.

## 3.2. Running the code

`python model.py` should run the code using `data_loader.py` as a module so, expect \_\_pycache\_\_ to appear. If there is an available GPU, the code will attempt to use it. The output printed on screen is the `sklearn.metrics.classification_report` calculated on the seen and unseen intents, separately.

`python model_run_all.py` runs an iteration over the list of all intents, leaving one as the unseen intent at each iteration and calculates the accuracies over the seen and unseen intents and wirtes them to "results.csv" file. This output is expected to be similar to the results in Table 2 in the paper, which is shown below.

## 3.3. Results
Table 1: The results taken from \[1\]
| Unseen Intent | Unseen Acc. | Seen Acc. |
| --- | --- | --- |
| AddToPlaylist | 0.9750 | 0.9500 |
| BookRestaurant | 0.9717 | 0.9600 |
| PlayMusic | 0.9683 | 0.8200 |
| RateBook | 0.9500 | 0.8800 |
| SearchCreativeWork | 0.8500 | 0.9200 |
| SearchScreeningEvent | 0.9867 | 0.7900 |
| GetWeather | 0.9733 | 0.9500 |

Table 2: The results of my experiments
| Unseen Intent | Unseen Acc. | Seen Acc. |
| --- | --- | --- |
| AddToPlaylist | 0.9194 | 0.9219 |
| BookRestaurant | 0.9565 | 0.8914 |
| PlayMusic | 0.8605 | 0.8730 |
| RateBook | 0.9875 | 0.9113 |
| SearchCreativeWork | 0.8879 | 0.9359 |
| SearchScreeningEvent | 0.8131 | 0.9224 |
| GetWeather | 0.9808 | 0.8993 |

Table 3: The difference between my experiment results and the results from \[1\]
| Unseen Intent | Unseen Acc. | Seen Acc. |
| --- | --- | --- |
| AddToPlaylist | -0.0556 | -0.0281 |
| BookRestaurant | -0.0152 | -0.0686 |
| PlayMusic | -0.1078 | 0.0530 |
| RateBook | 0.0375 | 0.0313 |
| SearchCreativeWork | 0.0379 | 0.0159 |
| SearchScreeningEvent | -0.1736 | 0.1324 |
| GetWeather | 0.0075 | -0.0507 |

# 4. Conclusion

The paper presents and approach to model intent classes in an intent space using RNN structures which is claimed to help zero-shot modeling of unseen intents. The classification architecture is quite dissimilar to what is customary in the literature and the training procedure calls for careful scheduling, although it is stated that no sophisticated parameter tuning is done, in the paper. With introduction of weight decay on the coordinate parameters of the model and use of SGD optimizer with early stopping, the model achieves accuracy levels close to those reported in the paper. 

As for my critisisms on the paper, the choice of accuracy as the evaluation metric is limiting. The results are reported over a single fold which raises questions about statistical significance of the reported results. There is no baseline method evaluated against the proposed approach. Use of a single dataset in the experiments (the results on the ATIS dataset are not provided in detail and not elaborated on rigorously) is also limiting the generalizability of the proposed method.

# 5. References

\[1\] [Jacobsen, S. A., & Ragni, A. (2021). Continuous representations of intents for dialogue systems. arXiv preprint arXiv:2105.03716.](https://arxiv.org/pdf/2105.03716.pdf)

\[2\] [Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).] (https://aclanthology.org/D14-1162.pdf)

# Contact

Eyüp Halit Yılmaz

yilmaz.eyup@metu.edu.tr

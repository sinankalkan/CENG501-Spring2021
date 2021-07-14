# Continuous-Representations-of-Intents-for-Dialogue-Systems

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper is a preprint available [here](https://arxiv.org/abs/2105.03716) that addresses the problem of representing user intents in conversational assistants. The main idea is to map every intent class to a point in space that is available through a set of coordinates and base matrices. In case of an encounter with an unseen intent, the model is able to represent it within the calculated space and classify it in the zero-shot learning setup.

## 1.1. Paper summary

The paper introduces the notion of intent space and proposes a new zero-shot detection algorithm. It uses recurrent neural network sturctures and an open-source [dataset](https://github.com/sonos/nlu-benchmark). The evaluation metric is accuracy and the results are reported after single fold evaluation.

# 2. The method and my interpretation

## 2.1. The original method

The original method is described in rather generic terms where an intent space with a set of bases and coordinates for each intent is able to represent every intent distinctly. In order for this parameter sets, bases and coordinates, to be learnable, the paper proposes to use an RNN structure and encode the base information within the hidden state multiplier of the RNN. Namely, if h is the hidden state of the RNN and U * h is the feed forward calculation for the RNN, U is the base matrix chosen for the related class. Bases of the space are taken as matrices to increase modeling power with use of more parameters.

## 2.2. My interpretation 

Based on the explanations in the paper, I assumed there exist several different RNN modules for each intent class available during training since the base for a given intent needs to be specific to that intent. The paper also states there exit different initial hidden vectors for each intent class, therefore, I assumed every RNN module uses a different initial hidden state vector (randomly initialized).

I assumed that the output of each RNN module is a base vector (I did not implement base matrices for the sake of simplicity) whose size is equal to the number of seen classes during training. This size constraint is also mentioned in the paper. In order to achieve this, a fully connected layer maps the 128 dimensional output to the number of seen intents (6 for the SNIPS dataset).

The obtained base matrices for a given input is multiplied by the coordinate matrix which is initialized as the identity matrix. During training, as explained in the paper, the base matrices are learned for the first 5 epochs and coordinates are frozen, after which base matrices are frozen and coordinates are learned  for another 5 epochs. Lastly, base matrices are updated again while coordinates are frozen for 5 epochs. This training procedure uses only seen intents during training.

After 15 epochs, the regularization term calculated over a small set of unseen intents is used to update the parameters of what is described as the expansion matrices in the paper. These matrices are of the same size with the base matrices and allow higher modeling power with introduction of extra parameters. During this second training phase, which lasts for another 15 epochs, bases and coordinates are frozen and only the expansion matrices are updated. I implemented an expansion matrix for every base although expansion matrices are defined only for unseen intents in the paper. In my implementation, every base matrix is multiplied by a distinct expansion matrix which increases modeling power even further.

The regularization term on the unseen intents described in the paper fails to train the model in my experiments, so I employed Cross Entropy Loss for the set of unseen intents as well.

# 3. Experiments and results

## 3.1. Experimental setup

SGD optimizer with learning rate 0.1 is used for the experiments. I observed Adam to be failing possibly due to the weight decay introduced on the coordinate parameters. At each epoch, early stopping is employed with 0.85 accuracy on the validation set, calculated every 1000 steps. The batch size is chosen as 1 for the experiments in order to simplify calculations of matrix multiplications. Weight decay on the coordinate parameters is chosen as 1e-5. The unseen intent is chosen as "GetWeather" as in the paper.

During training I observe a severe drop in validation accuracy at the start of 11th epoch, which happens after the bases are learned for 5 epochs and coordinates are learned in the following 5 epochs. For this 10-epoch period, validation accuracy increases monotonically but at the start of the 11th epoch, there is a dramatical drop. This is not reported in the paper since the reported accuracy values are those at the end of the epochs. The validation accuracy climbs back to its initial value by the end of the epoch in my experiments as well. Based o this observation, I abandoned the 15-epoch training schedule explained in the text of the paper and adopted the training schedule described in Figure 3 of the paper. The schedule is learning bases for the first 3 epochs, then learning coordinates for 2 epochs, then learning bases for 5 epochs and then learning only the expansion matrices for the rest of 30 epochs. After the epoch 7, unseen intents are used to calculate the regularization term.

## 3.2. Running the code

`python model.py` should run the code using data_loader.py as a module so, expect \_\_pycache\_\_ to appear. If there is an available GPU, the code will attempt to use it.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

Provide your references here.

# Contact

Provide your names & email addresses and any other info with which people can contact you.

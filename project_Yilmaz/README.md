# Continuous-Representations-of-Intents-for-Dialogue-Systems

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper is a preprint available [here](https://arxiv.org/abs/2105.03716) that addresses the problem of representing user intents in conversational assistants. The main idea is to map every intent class to a point in space that is available through a set of coordinates and base matrices. In case of an encounter with an unseen intent, the model is able to represent it within the calculated space and classify it in the zero-shot learning setup.

## 1.1. Paper summary

The paper introduces the notion of intent space and proposes a new zero-shot detection algorithm. It uses recurrent neural network sturctures and an open-source [dataset](https://github.com/sonos/nlu-benchmark). The evaluation metric is accuracy and the results are reported after 1-fold evaluation.

# 2. The method and my interpretation

## 2.1. The original method

The original method is described in rather generic terms where an intent space with a set of bases and coordinates for each intent is able to represent every intent distinctly. In order for this parameter sets, bases and coordinates, to be learnable, the paper proposes to use an RNN structure and encode the base information within the hidden state multiplier of the RNN. Namely, if h is the hidden state of the RNN and U * h is the feed forward calculation for the RNN, U is the base matrix chosen for the related class. Bases of the space are taken as matrices to increase modeling power with use of more parameters.

## 2.2. My interpretation 

Based on the explanations in the paper, I assumed there exist several different RNN modules for each intent class available during training since the base matrix for a given intent needs to be specific to that intent. The paper also states there exit different initial hidden vectors for each intent class, therefore, I assumed every RNN module uses a different initial hidden state vector (randomly initialized).

I assumed that the output of each RNN module is a single value which multiplies

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

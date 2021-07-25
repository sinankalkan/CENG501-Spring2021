# Semi-Supervised Learning under Class Distribution Mismatch

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

Our project paper is named "Semi-Supervised Learning under Class Distribution Mismatch". The paper whose authors are Yanbei Chen, Xiatian Zhu, Wei Li and Shaogang Gong was published at The Thirthy-Fourth AAAI Conference on Artificial Intelligence (AAAI-2020). The paper aims to apply semi-supervised learning (SSL) under the existence of a class distribution mismatch (CDM).

## 1.1. Paper summary

As the name suggests, the paper is about SSL. In ideal cases, we have small labelled data and large unlabelled data are drawn from the same class distribution. However, this may not be the case for most of the times. One may draw first part of the data from a class and second part from another class. This non-equality creates a mismatch while distributing the classes. There are different SSL methods that can be listed as but not limited to "temporal-ensembling", "mean-teacher", "pseudo-label" etc. And these methods suffer from a performance penalty under CDM. This penalty is caused by irrelevant unlaballed samples via error propagation. The authors in this paper suggests a novel method named Uncertainty-Aware Self-Distillation (UASD) method. This method combines Self-Distillation and Out-Of-Distribution (OOD) filtering methods.   
Self-distillation is distilling the knowledge from the deepest classifier to the shallower classifiers. This is done via attaching several attention modules and shallow classifiers at different dephts to a neural network. Knowledge-distillation on the other hand, is used to transfer the knowledge from an ensemble of many neural networks into a single network. This is done by averaging outputs of a whole ensemble of “teacher” networks and generating a soft target for a “student” network. 
Distribution in image classification has a bit different meaning than Language. Here, if we consider a cat playing with a ball who was thrown by a person, images/types of cats be in-distribution while images of balls, people etc. would be out-of-distribution. 
These approaches are implemented and modified together to create a novel approach: UASD. Its algorithm is given at Section 2.1 below. UASD is then trained on CIFAR10 & CIFAR100 datasets. The method is compared against baseline (supervised & using only labelled data for training), pseudo-label, pi-model, temporal-ensembling, mean-teacher, virtual-adversarial-training (VAT) and stochastic-weight-averaging (SWA) methods. In the paper, comparison is done many times to see performance at different labelled/unlaballed class mismatch ratios. When, the class mismatch ratio goes up, the error rate goes up as well. Also, different error rate for different mismatch ratios are shown for iterations

# 2. The method and our interpretation

## 2.1. The original method
Algorithm suggested by authors is given below. Note: All the algorithms, equations etc. are shared within a formal and algebraic notation in paper. However, for educative purposes, we put them in pseudo-form so it is easier to grasp the concept.
Require: Labelled data, Dl . Unlabelled data Du. A trainable neural network and a ramp-up weighting function, w.
```sh
for t=1 to maximum_epoch
	Refresh confidence threshold, pi_thr per epoch
	for k=1 to maximum_iteration_per_epoch
	Forward propagation to accumulate prediction, q, as every in-batch sample
	Apply out-of-distribution filtering
	Update network parameters with loss function
	end for
end for
```
Here, q is calculated as the average (1/t) of summation of predictions for every epoch.
OOD is applied via predictive confidence score, c_t. Let c_t be equal to maximum q value of a sample (here q also means soft target that is briefly explained in paper). 
Define a function f. Samples are filtered with this function. Lower confidence scores (c_t) mean higher predictive uncertainty
```sh
if c_t ≥ pi_thr then f=1 ; the sample is selected
else c_t ˂ pi_thr then f=0; the sample is rejected
```
Finally loss function, L, is defined as:
```sh
L= H(y_true) + w.f * H(q)
```
H is the standard supervised cross-entropy loss for ground truth (y_true) labels. Second “w.f*H” part is the UASD loss. w is the ramp-up weighting function, f is the OOD filter given above, H is again the cross-entropy loss but this time for q (model predictions rather than ground-truth).
## 2.2. Our interpretation 

The authors explain UASD very well both illustratively and algebraically, still algorithm and its sub-functions have to be coded to have the “on-the-fly” approach of the paper. 
# 3. Experiments and results

## 3.1. Experimental setup
We built our experiments upon open-source Tensorflow implementation by Oliver et al. (2018). Standard Wide ResNet (WRN-28-2, Zagoruyko and Komodakis 2016) is used as the base network. Adam (a first-order, gradient-based stochastic optimizadion method) optimiser is used for the training. Authors’ approach changed the default 10-dimensional classification layer to K-dimension where K is the number of known classes in labelled data so we modified that part as our framework as well. Rest of the parameters are same as Oliver et al.’s.
Paper datasets are CIFAR10, CIFAR100 and TinyImageNet. Oliver’s implementation for CIFAR10 & SVHN. However, we only achieved to implement UASD for CIFAR10 dataset.

## 3.2. Running the code
In order to run our code, you must have a folder that contains: “lib”, “runs”, “third_party”, “mnt” folders and “build_label_map”, “build_tfrecords”, “evaluate_checkpoints”, “evaluate_model”, “train_model” scripts. 
You can use “Benchmark & Studies” folder to access some of the links we used to learn about semi-supervised-learning and methods mentioned in the paper. 
We created “Cifar10” and “SVHN”  folders to understand Oliver’s setup. They contain output files of “build_label_map” and “build_tfrecords” scripts. They are not necessary to have and we put them just to understand how labeling etc. works while pre-processing the data and model. 
“Framework&Libraries” folder should be extracted to a folder named “lib” in order to work. We renamed it so that we know we are working on the framework and some libraries that this model is based on.
“runs” folder is same as the original. This folder contains many .yml files. They are first created by Oliver et al. to automatically run and evaluate the model with given verbosity, dataset name, root etc.
“OurApproach” folder contains “ssl_framework” and “train_model” files. “ssl_framework” script, as the name suggest, contains our framework for the ssl approach. This file should be extracted to and replaced within “lib” folder. We put it in this folder so that one can differentiate it with the original framework. Same for the “train_model”  script. It is to imitate UASD on CIFAR10 and this file should be replaced with the original file
“third_party” folder contains license and its utilisation with tensorflow.
After creating the folder with given sub-folders and scripts, dependencies must be checked and installed to run. You need: “ numpy” , “absl-py”, “tensorflow-gpu”, “scipy” & “tmuxp”. Oliver et al. didn't specify and hardware or software requirements other than these to work and you are not obliged to use any specific platform. But, we encountered many trial and errors during our work so we suggest:
1)	Ignore downloading the dependices as given in Oliver’s repo via pip3 install requirements.txt. Installing dependencies with pip3 for specific versions worked better. You may encounter version misfit or support problems. Python 3.7 with Tensorflow-Gpu 1.1x can be taken as a good starting point. If you want the newest tech, you should edit the code as well because code contains “logging” function of TF that doesn’t exist in TF 2.0
2)	We also used Tensorflow instead of Tensorflow-gpu. As expected, train times were longer but result didn’t change
3)	We ran the code within a virtual machine that runs Ubuntu 18.04. Different Linux distros or Windows etc. would probably change the results as not every OS support every version of dependencies. This is also given in Oliver’s repo as every version of Tensorflow may output different result.
4)	Due to write/read mechanism of temporary files, we had problems on Windows 10. There are some suggested solutions in the internet but they are cumbersome. It is not impossible to run on Windows however it may not worth the hassle. 
5)	The method contains writes & reads not only after executing scripts but also during the run-time. Thus, we suggest using “sudo” when permission or other type of r/w errors are encountered. This is not obligatory.


First, records and label should be constructed with these lines (we directly used terminal where directory is the folder that contains all folders)
```sh
python3 build_tfrecords.py --dataset_name=cifar10
python3 build_label_map.py --dataset_name=cifar10
```
Outputs of these are shared in our repo so check it if you wanna see results beforehand. Before moving on, by default, you should create a folder named “experiment-logs” under “mnt” folder. This is where all the outputs are written. However, if you want you can change it.
When output directory is ready, you should run line below:
```sh
tmuxp load runs/OURRUN.yml
```
Tmuxp is a terminal session manager that works like a multiplier. The authors automatically created these auto-run yml files. So if you want, you can even experiment with other datasets. 
Example for SVHN:
```sh
python3 build_tfrecords.py --dataset_name=svhn
python3 build_label_map.py --dataset_name=svhn
tmuxp load runs/figure-3-svhn-500-vat.yml
```

Here figure 3 means the 3rd figure in Oliver’s article. The article’s arxiv link is at references and we also put the paper in our repo in case you want to directly check it.
## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References
Instead of in-text references, we decided to go for this way as one can easily check internet for any subject.
Some parts of the base is subjected to Apache License 2.0. Therefore we claim no trademark usage but only educational modifications (https://www.apache.org/licenses/LICENSE-2.0). 
Oliver et al.’s repo for Tensorflow experiments: https://github.com/brain-research/realistic-ssl-evaluation
Oliver et al’s paper for SSL: https://arxiv.org/abs/1804.09170
Abseil’s documentary to understand flags as they are used exhaustively: https://abseil.io/docs/python/guides/flags

# Contact

Göksu Anıl KIZILDOĞAN - goksu.kizildogan@gmail.com (or just add me on https://www.linkedin.com/in/goksu-anil-kizildogan-b90aa3116/)

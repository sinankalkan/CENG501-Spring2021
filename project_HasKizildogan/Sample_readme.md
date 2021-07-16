# Semi-Supervised Learning under Class Distribution Mismatch

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

Our project paper is named "Semi-Supervised Learning under Class Distribution Mismatch". The paper whose authors are Yanbei Chen, Xiatian Zhu, Wei Li and Shaogang Gong was published at The Thirthy-Fourth AAAI Conference on Artificial Intelligence (AAAI-2020). The paper aims to apply semi-supervised learning (SSL) under the existence of a class distribution mismatch.

## 1.1. Paper summary

As the name suggests, the paper is about SSL. In ideal cases, we have small labelled data and large unlabelled data are drawn from the same class distribution. However, this may not be the case for most of the times. One may draw first part of the data from a class and second part from another class. This non-equality creates a mismatch while distributing the classes. There are different SSL methods that can be listed as but not limited to "temporal-ensembling", "mean-teacher", "pseudo-label" etc. And these methods suffer from a performance penalty under CDM. This penalty is caused by irrelevant unlaballed samples via error propagation. The authors in this paper suggests a novel method named Uncertainty-Aware Self-Distillation (UASD) method. This method combines Self-Distillation and Out-Of-Distribution (OOD) filtering methods.   

# 2. The method and my interpretation

## 2.1. The original method

Explain the original method.

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

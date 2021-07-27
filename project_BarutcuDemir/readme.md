# Beyond Fully Connected Layers with Quaternions: Parametrization of Hypercomplex Multiplications with 1/n Parameters

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

[This paper](https://openreview.net/pdf?id=rcQdycl0zyk) was published as a conference paper at ICLR 2021 by Zhang et al. It was among the 8 Outstanding Papers selected by ICLR organizers out of the 860 papers submitted to the conference.

Although the paper's title is intimidating, the main idea behind it is easy to understand. It builds up on top of the work by Parcollet ([2018](https://hal.archives-ouvertes.fr/hal-02107611/document), [2019](https://core.ac.uk/download/pdf/217859026.pdf)) and Tay ([2019](https://arxiv.org/pdf/1906.04393.pdf)). These papers propose various quaternion networks, which greatly reduce the number of parameters while attaining similar performances with their ordinary counterparts. For context, quanternions are an extension for the vanilla complex numbers, a + bi, which are in the form of, a + bi + cj + dk. One particular area where they are utilized to a great extent is computer graphics, where they can be used to model rotations. In machine learning they are used because the Hamilton product used to multiply two quaternions, serves as a great tool for represantation learning. Hamilton product is defined as ... But the problem with quaternion networks is that they can only be used to parametrize certain dimension sizes (4, 8, 12). This limits their potential, which brings us to the said paper.

## 1.1. Paper summary

Summarize the paper, the method & its contributions in relation with the existing literature.

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

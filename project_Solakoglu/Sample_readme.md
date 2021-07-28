# Using noise to probe recurrent neural network structure and prune synapses

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

In this project, I tried to reproduce the results of “Using noise to probe recurrent neural network structure and prune synapses” which was published at  NeurIPS 2020 [1].

## 1.1. Paper summary

During the lifetime of human brain, its’ synapses can eliminated or added onto dendritic branch via different plasticity methods. In healthy brain this process improves brain’s capacity of perception, memory, behaviour etc [2-4]. The mammalian brain can do organize its’ synapses very efficient by spending ~20 W of energy. However state-of-the-art machine learning (ML) algorithms mostly requires large amount of energy to mimic this process. Yet most of these algorithms still have different pitfall such as failing to transfer, catastrophic forgetting [5]. To reduce the energy requirement, one can implement synaptic elimation into ML algorithms. However, it is not clear that which synapses would be eliminated or added during neuronal activity.
Additionally, we know that biological systems are very noisy that can affect neuronal and cognitive outputs [6]. Addition of noise would improve the realism of neuron models and ML algorithms [7]. Accordingly, Moore and Chaudhury inserted noise into their neuron model to structure and prune synapses.

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

1.	https://arxiv.org/abs/2011.07334
2.	El-Boustani S, Ip JPK, Breton-Provencher V, Knott GW, Okuno H, Bito H, Sur M. Locally coordinated synaptic plasticity of visual cortex neurons in vivo. Science. 2018 Jun 22;360(6395):1349-1354.
3.	Lai CS, Franke TF, Gan WB. Opposite effects of fear conditioning and extinction on dendritic spine remodelling. Nature. 2012 Feb 19;483(7387):87-91.
4.	Yasuda R. Biophysics of Biochemical Signaling in Dendritic Spines: Implications in Synaptic Plasticity. Biophys J. 2017 Nov 21;113(10):2152-2159.
5.	Chavlis S, Poirazi P. Drawing inspiration from biological dendrites to empower artificial neural networks. Curr Opin Neurobiol. 2021 Jun 1;70:1-10.
6.	Faisal AA, Selen LP, Wolpert DM. Noise in the nervous system. Nat Rev Neurosci. 2008 Apr;9(4):292-303.
7.	Pulvermüller F, Tomasello R, Henningsen-Schomers MR, Wennekers T. Biological constraints on neural network models of cognitive function. Nat Rev Neurosci. 2021 Aug;22(8):488-502. 

# Contact

You can contact me via sabahaddin.solakoglu@hacettepe.edu.tr
S. Taha Solakoglu

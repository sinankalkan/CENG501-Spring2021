# Using noise to probe recurrent neural network structure and prune synapses

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

In this project, I tried to reproduce the results of “Using noise to probe recurrent neural network structure and prune synapses” which was published at  NeurIPS 2020 [1].

## 1.1. Paper summary

During the lifetime of human brain, its’ synapses can eliminated or added onto dendritic branch via different plasticity methods. In healthy brain this process improves brain’s capacity of perception, memory, behaviour etc [2-4]. The mammalian brain can do organize its’ synapses very efficient by spending ~20 W of energy. However state-of-the-art machine learning (ML) algorithms mostly requires large amount of energy to mimic this process. Yet most of these algorithms still have different pitfall such as failing to transfer, catastrophic forgetting [5]. To reduce the energy requirement, one can implement synaptic elimation into ML algorithms. However, it is not clear that which synapses would be eliminated or added during neuronal activity.

Additionally, we know biological systems are very noisy that can affect neuronal and cognitive outputs [6]. Addition of noise would improve the realism of neuron models and ML algorithms [7]. Accordingly, Moore and Chaudhury inserted noise into their neuron model to structure and prune synapses.

# 2. The method and my interpretation

## 2.1. The original method

Moore and Chaudhury started to construct their model by forming a linear neural network of the form:

![image](https://user-images.githubusercontent.com/47305046/127342976-49ee1493-7c5f-4fd1-89af-1dd8f4bed322.png)

![image](https://user-images.githubusercontent.com/47305046/127343023-2b94ebfe-962e-49cf-9474-4a1d6c685ff2.png)

The vector *x* represents the firing rate of *N* neurons with *x<sub>i</sub>* is the firing rate of the *i*-th neuron. *b*(t) is the external input to the neuron that is considered as noise, *b* is an arbitrary vector of constant background input to the network and ξ is a vector of IID unit variancer Gaussian white noise and *σ* is the standard deviation of the input noise. *D* is a diagonal martix representing the intrinsic leak of activity. *W* is the matrix of weighted connections between the neurons with wij is the connection strength from the *j*-th neuron to *i*-th neuron. Then they defined a matrix *A*:
 
![image](https://user-images.githubusercontent.com/47305046/127343074-02de2f20-ea23-49c0-be5b-2f19b6c6d035.png)

Their pruning rule seek to generate a sparse network with corresponding matrix Asparse with two properties:
  
  1-	Small number of edges (i.e. number of non-zero entries in *A<sup>sparse</sup>*) to create different neuron groups
  
  2-	Dynamics of this network seeks to be similar to dynamics of original network:
  
  ![image](https://user-images.githubusercontent.com/47305046/127343192-cb8de988-f943-49a1-be30-f9999ef1a7b1.png)
 
This sparse network has made similar by adaptaiton of spectral similarity from graph sparsification [8,9]: 

![image](https://user-images.githubusercontent.com/47305046/127343322-18f2e3f0-66fb-4b44-8791-10f624591e0e.png)

for some small *Ɛ* > 0. This rule helps to preserve eigenvalues, eigenvectors of matrix *A*.

Then they activated neurons and calculated pruning probability of neuronal connection:

![image](https://user-images.githubusercontent.com/47305046/127343433-886f4127-b7dc-41a1-9e6e-5d86d4f8b0ca.png)

*C* is the covariance matrix of firing rates in response to white noise input, *C_<sub>ii</sub>* and *C_<sub>jj</sub>* are the variances of *i*-th and *j*-th neurons and *C_<sub>ij</sub>*  are their covariance. *K* is a proportionality constant. Finally they determine *A<sup>sparse</sup>* for *i* ≠ *j*:

![image](https://user-images.githubusercontent.com/47305046/127343569-4dc14d0f-65a0-4967-844b-2909ec13e38d.png)
 
For the diagonal terms, they preserved diagonal terms or defined a minimal perturbaiton as,

![image](https://user-images.githubusercontent.com/47305046/127343627-b3dd2892-e98b-4d1f-b3c8-5f03c8916a35.png)

![image](https://user-images.githubusercontent.com/47305046/127343648-be7b3a28-43ac-4750-82b8-cc35d09208ec.png)

Finally, they showed how they obtained their formulas and and obtained numerical results for 3,000 and 10,000 neurons. Finally they described some extensins of their study.

## 2.2. My interpretation 

In the introduction of this study. Authors mentioned how synaptic connection change their connectivity by adding or pruning synapses. However synaptic implementation to this model ended with introduction and they defined the connectivity as classical neural network connectivity. Therefore study has become limited with a new neural network type that changes connectivity between neurons according to recently defined rule.

At numerical analysis authors well defined connectivity structure. Therefore I could created W, D and external noisy input. However authors could not difine firing rate vectors and how to obtain covariance matrix structure since obtaining covariance matrix with size of N x N was not possble with firing rate at 0 s. Threfore I used Eliphant to create firing rate covariance rate at 0 s [10].

# 3. Experiments and results

## 3.1. Experimental setup

In the original paper, authors have done their numerical analysis for 3,000 neurons (with clusters of 2,700, 100, 100 and 100 neurons) and 10,000 neurons (with 10 clusters of 100 neurons and 1 cluster of 9,000 neurons). I started to create my algorithm with the first selection (3,000) neurons.

Authors defined two types of connection to create connectiviy matrix:

 1-	Dense within-cluster connections (%60 of connections with the weight of ~ *N*(1,1))
 
 2-	Sparse long-range connections (5000 total with the weight of ~ *U*(0,1))

I created the connectivity matrix by using *within_network*, *sparse_network* and *connectivity_matrix* algorithms according to above rule.

Next, authors defined external input by assigning *b* as 0.0002 and *ξ* as a gaussian white noise. In the original paper authors defined *x*(0) (i.e. firing rates at 0 s) as uniform random entries with a rule of ~*U*(0,1). But this definition was not enough to create covariance matrix. Therefore I used *homogenous_poisson_process* and *instantaneous_rate algorithms* form Elephant library [10]. This code gives a firing rate with Poisson process with a defined firing rate and sampling interval and period. To get an easy example, I created firing rates 3,000 neurons at 10 Hz with Poisson distribution for 10 seconds with sampling period of 50 ms. The leaking activity of these neurons was obtained with *get_leaking* algorithm.

When I created the structure of my network. I obtained Asparse via get_probs and a_sparse algorithms. After obtaining *A<sup>sparse</sup>*, I have chosen to protect diagonal structure via *preserve_diag algorithm* since authors have showed whether preseving or not preserving does not change the results. Matrix multiplication was not supported in Elephant due to Quantities library does not supporting matrix multiplication. To overcome this issue I have created *matrix_multip* algorithm that makes just basic matrix operation. Finally I have tried my algorithms to work via dxdt algorithm.

## 3.2. Running the code

My code can basically run by anyone who has simple computer. It does not require a GPU.

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
8.	Spielman, D. A. & Srivastava, N. Graph sparsification by effective resistances. SIAM Journal on Computing 40, 1913–1926 (2011).
9.	Spielman, D. A. & Teng, S.-H. Spectral sparsification of graphs. SIAM Journal on Computing 40, 981–1025 (2011).
10. Denker M, Yegenoglu A, Grün S (2018) Collaborative HPC-enabled workflows on the HBP Collaboratory using the Elephant framework. Neuroinformatics 2018, P19


# Contact

You can contact me via sabahaddin.solakoglu@hacettepe.edu.tr
S. Taha Solakoglu

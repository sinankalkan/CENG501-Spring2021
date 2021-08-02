# Approximating Ground State Energies and Wave Functions of Physical Systems with Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# Disclaimer
The structure of the code is inspired from the CENG501 course materials.  

# 1. Introduction
This study was presented at the Machine Learning and the Physical Sciences Workshop at the 34th Conference on Neural Information Processing Systems (NeurIPS) on 
December, 2020. It aims at approximating the ground state energies and eigenfunctions of quantum mechanical sytems via Neural Networks. My goal was to obtain the 
same result with the original paper.    

## 1.1. Paper summary

In Quantum Mechanics, the solution of the Schrödinger's equation provides full knowledge of the dynamics of physical systems. However, there might be complex systems 
which may not be solved analytically such as many body systems like Helium atom. There are several methods to retrieve the energy eigenvalues of such systems such as 
variational methods. Once the problem is reduced to an optimization problem one can use Neural Networks to solve the problem. In the literature, it is often to see 
the methods which uses trial wavefunctions and imposed boundary conditions, then seeks for the solution. In this paper, however, there is no trial wavefunction and 
boundary conditions. The problem was solved in an unsupervised manner. The method is described in more details in section 2.1 

# 2. The method and my interpretation

## 2.1. The original method

The method that was used in the study was straightforward. Given the basis functions of the system, the expectation value of the Hamiltonian was used as the loss 
function. The input of the network is the position and the output is the ground state wave function.    

## 2.2. My interpretation 
Some of hyperparameters such as the learning rate and the number of epochs were not expressed in the study. Therefore, it was cumbersome to finetune the network.
I trained the networks with large number of epochs and small learning rates. After several attempts for finetuning, the network started to give similar results with 
the original work. The loss definitions and the training procedure can be found in Section 3.1 
  
# 3. Experiments and results

## 3.1. Experimental setup

In the study, the one dimensional infinite well with and without perturbation were inspected. For the unperturbed system with the well width a = 1 , a neural 
network with one hidden layer of 1000 ReLU activations were used. However, in my experiments I found this not robust and put one extra hidden with 1000 ReLU 
activations. The basis functions in the one dimensional infinite well were defined as [[1]](#1),    

![b_n](https://user-images.githubusercontent.com/47567854/127366652-068e95a8-2377-4726-8aaf-7dfb3a20b37b.png)

For every experiment N = 100 basis functions were used. The expectation value of the Hamiltonian in the unperturbed system, i.e. the loss function is given as 
[[2]](#1),    

![hamilton_unperturbed](https://user-images.githubusercontent.com/47567854/127367205-7dd102ba-17a6-477d-85c7-7227144211a2.png)

where the energy eigenvalues of the unperturbed system is [[3]](#1),

![render_2](https://user-images.githubusercontent.com/47567854/127366875-5f90533f-07f8-4eb0-8f5a-ed42c5f02176.png)

Here, a is the well width and μ is the mass of the particle which is taken to be 1 in each experiment. ħ is the Planck's constant in natural units.
As far as the perturbed system is concerned, a perturbation term was added to the unperturbed Hamiltonian i.e. 

![add_hams](https://user-images.githubusercontent.com/47567854/127367931-2d522c02-297b-43f6-930f-9c9f88650724.png)

where the perturbed Hamiltonian can be defined as [[4]](#1), 

![perturbed_hamiltonian](https://user-images.githubusercontent.com/47567854/127368324-5fd3d8e6-a475-4c08-8d5b-f0832a54e9b6.png)

α is the perturbation constant and where [[5]](#1),

![x_nm](https://user-images.githubusercontent.com/47567854/127368545-55bb0f0a-e94a-4272-beec-632f193427b5.png)

For the perturbed setup a network with two hidden layers with 500 and 100 ReLU activations were used for two configurations,

i. a=1 & α = 8.

ii. a = 10 & α = 2.

Theoretically, the input space (i.e. the position space) must be continous. It was stated that the projections of the outputs (i.e the wave functions) onto the 
Hilbert space (i.e. the basis functions) were transformed into discrete values and hence in each setup the decompositions of wave functions were computed using 
Riemann's sum,  

![riman](https://user-images.githubusercontent.com/47567854/127406137-83a344c3-a22b-49da-9935-a1cc3c6b8961.png)

## 3.2. Running the code

I prepared the code on Google's colab framework. One can easily follow the relevant cells.  

## 3.3. Results
The ground state wave functions obtained by the original work can found be in Figure 1 (figure adapted from [[6]](#1) ). Moreover, in the table below, I compared the 
energy eigenvalues of the the original study and those of mine. In the perturbed systems, the finetuning requires more efforts.  

![fig](https://user-images.githubusercontent.com/47567854/127409056-47dea50d-6179-4d6c-9aa8-d83adbb63c23.png)

System | Original Work | Exact | Mine 
--- | --- | --- | ---
Unperturbed | 4.93484 | 4.9348  | 4.9711 
Perturbed A | 8.79510 | 8.79507 | 8.9710 
Perturbed B | 2.94583 | 2.94583 | NA 

In Figure 2 the loss curves I obtained during the training are presented. The perturbed system learns the ground state quickly whereas Unperturbed system, in 
particular, system ii oscillates.

![losses_all](https://user-images.githubusercontent.com/47567854/127466021-c1d0b6b2-1453-4ab6-95f6-39298892026f.png)
**Figure 2:** Loss curves for three systems. Perturbed systems requires more finetuning.

Finally , the ground state energies I obtained are presented in Figure 3.
![wf_all](https://user-images.githubusercontent.com/47567854/127467107-ad79bdbd-3501-4dfb-8020-e61d39442ffe.png)
**Figure 3:** Ground state wave functions. Perturbed system requires more finetuning

# 4. Conclusion
In the original work they reached the true values with very high accuricies. On the other hand, the ground state energies and the shape of the ground state 
wavefunctions I found were close to the those of the original paper. Apart from unperturbed system ii, there was some fractional differences between the wave 
functions which is probably due to normalizations. 

# 5. References

<a id="1">[1]</a> 
Lema, C & Choromanska, A.(2020). 
Approximating Ground State Energies and Wave Functions of Physical Systems with Neural Networks.

# Contact
Onur Durhan <br />
Middle East Technical University, Department of Physics <br />
email: onur.durhan@metu.edu.tr <br />

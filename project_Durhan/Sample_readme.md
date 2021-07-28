# Approximating Ground State Energies and Wave Functions of Physical Systems with Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction
This study was presented at the Machine Learning and the Physical Sciences Workshop at the 34th Conference on Neural Information Processing Systems (NeurIPS) on 
December, 2020. It aims at approximating the ground state energies and eigenfunctions of quantum mechanical sytems via Neural Networks. Our goal was to obtain the 
same result with the original paper.    
Introduce the paper (inc. where it is published) and describe your goal (reproducibility). 

## 1.1. Paper summary

In Quantum Mechanics, the solution of the Schrödinger's equation provides full knowledge of the dynamics of physical systems. However, there might be complex systems 
which may not be solved analytically such as energy levels of Helium atom. There are numerical methods to retrieve the eigenenergies of physical systems such as 
variational methods. Once the problem is reduced to a optimization problem one can use Neural Networks to solve the problem. In the literature, it is often to see the 
methods which uses trial wavefunctions and imposed boundary conditions than seek for the solutions. In this paper, however, there is no trial wavefunctions and 
boundary conditions. The problem was solved in an unsupervised manner. The method is described in more details in section 2.1 

Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

The method that was used in the study was quite simple. Given the basis functions of the system, the expectation value of the Hamiltonian was used as the loss 
function. The input of the network is the position and the output is the ground state wavefunction.    

Explain the original method.

## 2.2. My interpretation 
Some of hyperparameters such as the learning rate and the number of epochs were not expressed in the study. Therefore it was cumbersome to finetune the networks. 

 were calculated using Riemann's sum. According to this, one has,

The loss definitions and the training can be found in section 3.1 
  
Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

In the study the one dimensional infinite well with and without perturbation were inspected. For the unperturbed system with the well width a = 1 , a neural 
network with one hidden layer of 1000 thousands ReLU activations were used. However, in my experiments I found this insufficient and put one extra hidden layer of the 
same size. The basis functions in the one dimensional infinite well were defined as [[1]](#1), ,    

![b_n](https://user-images.githubusercontent.com/47567854/127366652-068e95a8-2377-4726-8aaf-7dfb3a20b37b.png)

For every experiment N = 100 basis functions were used. The expectation value of the Hamiltonian in the unperturbed system, i.e. the loss function is given as 
[[2]](#1), ,     

![hamilton_unperturbed](https://user-images.githubusercontent.com/47567854/127367205-7dd102ba-17a6-477d-85c7-7227144211a2.png)

where the energy eigenvalues of the unperturbed system is [[3]](#1), ,

![render_2](https://user-images.githubusercontent.com/47567854/127366875-5f90533f-07f8-4eb0-8f5a-ed42c5f02176.png)

As far as the perturbed system is concerned, a perturbation term should be added to the unperturbed Hamiltonian i.e. 

![add_hams](https://user-images.githubusercontent.com/47567854/127367931-2d522c02-297b-43f6-930f-9c9f88650724.png)

where the perturbed Hamiltonian can be defined as [[4]](#1), , 


![perturbed_hamiltonian](https://user-images.githubusercontent.com/47567854/127368324-5fd3d8e6-a475-4c08-8d5b-f0832a54e9b6.png)

α is the perturbation constant and where [[5]](#1),

![x_nm](https://user-images.githubusercontent.com/47567854/127368545-55bb0f0a-e94a-4272-beec-632f193427b5.png)

For the perturbed setup a network with two hidden layers with 500 and 100 ReLU activations were used for two configurations,

i. a=1 & α = 8.

ii. a = 10 & α = 2.

Theoretically, the input space (i.e. the position space) must be continous. It was stated that the projections of the outputs (i.e the wave functions) onto the 
Hilbert space (i.e. the basis functions) were transformed to discrete numbers and hence in each setup the decompositions of wave functions were computed using 
Riemann's sum,  

![riman](https://user-images.githubusercontent.com/47567854/127406137-83a344c3-a22b-49da-9935-a1cc3c6b8961.png)


Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

I prepared the code on Google's colab framework. One can easily follow the relevant cells.   

Explain your code & directory structure and how other people can run it.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

The ground state energies I found were close to the those of original paper. The shape of the ground state wavefunctions were also quite similar to the those of the 
original ones. There was only fractional differences between the original

Discuss the paper in relation to the results in the paper and your results. 

# 5. References

<a id="1">[1]</a> 
Lema, C & Choromanska, A.(2020). 
Approximating Ground State Energies and Wave Functions of Physical Systems with Neural Networks.


Provide your references here.

# Contact

Provide your names & email addresses and any other info with which people can contact you.

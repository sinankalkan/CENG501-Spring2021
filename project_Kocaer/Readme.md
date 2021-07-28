# STAR: Sparse Trained Articulated Human Body Regressor

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

Star is a 3D human body pose and shape prediction model. This article was published at the ECVV symposium in 2020. The purpose of this project is to provide an application of the article "STAR: Sparse Trained Articulated Human Body Regressor" published in this symposium.

## 1.1. Paper summary

SMPL is a prediction model for 3D human pose and shape. Since SMPL has too many parameters, STAR was introduced by Osman et al. They found that when trained on the same data as SMPL, STAR generalized better despite having fewer parameters. Per-joint pose correctives are defined and a subset of mesh vertices affected by each joint movement is learned. Thus, the model parameters decreased to 20% of the SMPL. 

# 2. The method and my interpretation

## 2.1. The original method

STAR is a vertex-based LBS model complemented with a learned set of shape and pose corrective functions. In STAR, a pose correcting function is defined for each j-joint in the kinematic tree. In the STAR model, unlike the SMPL model, the non-linear activation function for each joint is used throughout the train. In addition, the pose corrective function is used to make a better estimation of the human body.

STAR parameters;

Pose Parameters: θ ∈ R^72 

Axis angle rotations joints: 24

Axis angle rotations shape: β ∈ R^10

The number of mesh vertices: 6890


## 2.2. My interpretation 

In the article, the where L1 norm and L2 norm loss functions will be applied in the training section are not entirely clear. When the functions are examined, it is seen that the L1 norm function should be applied in the quaternion() and rodrigues() methods. L2 norm function is applied in the loss section and total loss is applied in the trainer section.

# 3. Experiments and results

## 3.1. Experimental setup

I built my experiments upon Pytorch implementation by Osman et.al (2020). The .npz files required first to create the model are downloaded from https://star.is.tue.mpg.de/. The model includes extracting dicts from the .npz file and applying Linear Blend Skinning with forward(). In the article, CAESAR and SizeUSA datasets are used to train the model. However, access to these data sets was not available. 

## 3.2. Running the code

Code is implemented and can be run on Google Colab.

## 3.3. Results

Since the CAESAR and SizeUSA datasets applied in the article are not accessible (not free), the model could not be tested with these datasets. Therefore, a comparison on the results could not be made. I could not draw the heatmap pylot for the human body obtained in the article. But I got human body shape using 3D pylot feature of python.

# 4. Conclusion

The article presents 3D human body estimation as vertex based Linear Blend Skinning model. In the article, images obtained from the STAR model developed differently from SMPL and SMPL are available. Thanks to the pose corrector function in the STAR model, the cramping in the elbow bend is prevented.
The mean absolute error was found by training the model on two different data sets. With the results obtained, it was seen that the model was compatible with the data sets.


# 5. References

[1] Osman, A.A.A., Bolkart, T., Black, M.J., 2020. STAR: Sparse Trainde Articulated Human Body Regressor. ECVV 2020, 598-613. 

# Contact

Emine Rumeysa Kocaer
eminekocaer@sdu.edu.tr


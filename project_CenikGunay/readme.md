# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper presents **Placepedia**, a comprehensive place dataset that contains images for places of interest from all over the world with massive attributes, where it is published in _ECCV 2020._ The authors of the paper developed **PlaceNet**, a unified framework for multi-level place recognition. It simultaneously predicts place item, category, function, city and country. In the image below, you can see the hierarchical structure of Placepedia with places from all over the world. 

![Placepedia](https://user-images.githubusercontent.com/53267971/127356409-e9169861-a7ca-4c34-9d3e-5233e668ef4b.PNG)


## 1.1. Paper summary

We can summarize the paper as below.

**1)** They build Placepedia, a large-scale place dataset with comprehensive annotations in multiple aspects. Compared to the other dataset related to place recognition tasks, Placepedia is the largest and the most comprehensive dataset for place understanding as you can see from the Table 1 below, which shows the comparision with the other dataset.

**2)** Authors designed four task-spesific benchmarks on Placepedia with respect to multi-faceted information.

**3)** They conduct systematic studies on place recognition, which demonstrate important challenges in place understanding as well as the connections between the visual characteristics and underlying socioeconomic or cultural implications.

In the related studies, the dataset focuses only the spesific tasks like place categorization, scene recognition, object retrieval or image retrieval. However, Placepedia can be used for all these task due to its much larger amount image and data, containing over 240K places with 35 million images labeled with 3K categories. Moreover, the function information of places may lead to a new task, namely place functio recognition.

_**Table 1.** Comparing Placepedia with other existing datasets. Placepedia offers the_
largest number of images and the richest information
![comparision](https://user-images.githubusercontent.com/53267971/127357466-9ba97ed1-2ead-4646-8c25-17514200bd3f.PNG)


# 2. The method and my interpretation

## 2.1. The original method

They construct a CNN-based model to predict all tasks simultaneously. The training procedure performs in an iterative manner and the system is learned **end-to-end** learning. 

_**Network Structures**_

- The network structure of PlaceNet is similar to ResNet50. The only difference is that the last convolution/pooling/fc layers are duplicated to five branches, namely, place, category, function, city and country. Each branch contains two FC layers. 

_**Dataset-Placepedia**_

- Places-Coarse. They select 200 places for validation and 400 places for testing,
from 50 famous cities of 34 countries. The remained 25K places are used
for training. For validation/testing set, they double checked the annotation
results. Places without category labels are manually annotated. After merging
similar items of labels, they obtain 50 categories and 10 functions. The training/validation/testing set have 5M/60K/120K images respectively, from 7K
cities of more than 200 countries.
- Places-Fine. Places-Fine shares the same validation/testing set with PlacesCoarse. For training set, they selected 400 places from the 50 cities of validation/testing places. Different from Places-Coarse, they also double checked
the annotation of training data. The training/validation/testing set have
110K/60K/120K images respectively, which are tagged with 50 categories, 10
functions, 50 cities, and 34 countries. 

- After we downloaded the dataset Placepedia we get the following folders and files that used in our implementations.

**i. train:** includes the places for training.

**ii & iii.** valg+valq: includes the places for validation for tasks including place categorization, function categorization, and city/country recognition. For place retrieval task, valg includes the gallery images of validation places and valq includes the query images of validation places.

**iv & v.** testg+testq: includes the places for testing for tasks including place categorization, function categorization, and city/country recognition. For place retrieval task, testg includes the gallery images of testing places and testq includes the query images of testing places.

Four Category Id Files:

**i.** type_to_id.json: for place categorization task. Each category is associated with an id.

**ii.** function_to_id.json: for function categorization task. Each function is associated with an id.

**iii.** city_to_id.json: for city recognition task. Each city is associated with an id.

**iv.** country_to_id.json: for country recognition task. Each country is associated with an id.

_**PlaceNet**_

- Below you can see the Pipeline of PlaceNet.

![PlaceNetPipeline](https://user-images.githubusercontent.com/53267971/127476768-47dbd6bc-b209-47da-888c-218609dfa2aa.PNG)

   _**Figure 2.** Pipeline of Placenet_


- **Loss Functions**. They study three losses for PlaceNet, namely, **softmax loss**, **focal
loss**, and **triplet loss**. 


- **Pooling Methods**. They also study different pooling methods for PlaceNet,
namely, **average pooling**, **max pooling**, **spatial pyramid pooling**



## 2.2. My interpretation 


The biggest problem we encountered while implementing the solution proposed in the paper was that the data set was quite large. Also, due to the complexity of the model due to duplication for multiple predictions and hardware constraint, results corresponding to training only a part of the dataset will be presented here. Moreover, the learning rate 0.5 value given in the paper was not suitable as it is too big. That's why we chose the learning rate as 0.0001. All details will be given in the experimental setup.
Since the batch size is not specified in the training details section of the paper, we have set the batch size as 64. As we observe from the paper, loss functions focal and softmax give almost the same results. Therefore, we only use softmax loss function as well as average pooling. The other problem we encounter is related to dataset image shape. After starting training, after the some point we got the error **output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]**, which is due to the mismatch of grayscale image to RGB image,which means there is some grayscale images in the dataset and produces an error. Therefore, we have trained our model until we encountered this error without checking whole dataset since it has 240K such images. Our aim is to show that our model works correct and it tends to be more powerful compared to other datasets and models even if we can only train our model such constraints and problems.



# 3. Experiments and results

## 3.1. Experimental setup


In original setup authors use Placepedia dataset. Because of lack of disk space in Colab at this part we could only implement one tenth of it to some extent. Moreover, authors provided that they trained the network for 100K iterations which we couldn't train for that long as Colab limits gpu usage so we have presented the results for at most 700 iterations, where iteration training which is significantly lower than the author's setup. All parameters are listed in the table 2 below. 
  
  _**Table 2.** Training Parameters of our implementation and authors_
  
  ![parameter_tables](https://user-images.githubusercontent.com/53267971/127470144-8fc717cd-cd5e-44a3-98bd-bfae6c551b29.PNG)

  
  
## 3.2. Running the code

Readers can easily follow our implemetation given the folder below from the ``` PlaceNet_&_Placepedia.ipynb ``` which gives detailed explanation of the implementations.

```
project_CenikGunay
│  loss_graphs
│  readme.md 
│  PlaceNet_&_Placepedia.ipynb
  

```

* *PlaceNet_&_Placepedia.ipynb* notebook is used for all implemetation used in this report.
*  *loss_graphs* is generated by how the model PlaceNet works in spite of several constraints and problems.

## 3.3. Results


_**Table 3.** Comparision of our result with the previous works_
![Result_tables](https://user-images.githubusercontent.com/53267971/127472734-2f81fac0-d852-42ef-bbdc-7b886c5d43c8.PNG)

In this report, while we reproduce the results given in the paper, what we are trying to show is that the model performs well. As the number of epochs given in the paper is 90, but hardware limitation and many of the images in the dataset are not suitable for training. The training operation is finished without entering the range converged by the loss function, since only one epoch and a limited number of iterations are adhered to. You can see the tendency of the loss curve in 700 iteration with 1 epoch, where it takes more than 1 hour.Therefore, accuracy may seem low at first glance. However, we see that we are still able to approach some performances of other models in the first epoch. This gives an idea about the power of our implementation and the model.


![Loss_curve](https://user-images.githubusercontent.com/53267971/127474514-18f123dd-055e-4ca4-9460-67168a87170e.PNG)

_**Figure 3.** Loss curve with 700 iterations with 1 epoch_

# 4. Conclusion

In this work, we try to reproduce the results given in **[3]** by using a large-scale place dataset which is comprehensively
annotated with multiple aspects. We observed that it is the largest place-related
dataset available. To explore place understanding and correct implementations of the solution proposed, we carefully build our model based on the spesifications.Model PlaceNet is very similar to the Resnet50. The only difference is we did in the model and its implementation is multi level prediction. For this purpose, as a proposed solution, we duplicate the last convolution layer of Resnet50 to five branches. Each branch consists of 2 fc layers. All these implementations can be easily followed in ``` PlaceNet_&_Placepedia.ipynb ```. Our experimental experience show us that it is indeed a very complex problem since it is a very difficult task to train such a large data set and make the necessary adjustments.However, we believe that we use all the information we have learned in accordance with the course content and that the effort we spend is not bad compared to the results obtained and the complexity of the problem. The accuracy values obtained for Top1 and Top 5 show an approximation to the accuracy values of other models even in epoch 1 and 700 iterations. This gave us clues that the 90 epoch value PlaceNet model recommended in the paper would give the best performance, and we tried to clearly report this in the report.At the same time, considering the complexity of the problem during training, we determined some parameters that were not mentioned in the paper.However, we did not deviate from the general solution path. We think that the PlaceNet model was created in accordance with the specifications. Our report can be easily followed on the notebook and we see it as an advantage that it is both clear and simple.We think we learned a lot within the scope of the project. In this context, we would like to thank Sinan Hoca and the authors of the paper, who gave us the excitement of making the first paper implementation of our academic life.   




# 5. References

**[1]** Project Page, https://hahehi.github.io/placepedia.html

**[2]** Github Repo, https://github.com/hahehi/placepedia

**[3]** Placepedia: Comprehensive Place Understanding with Multi-Faceted Annotations,https://arxiv.org/pdf/2007.03777.pdf

# Contact

Beril Günay (beril.gunay@metu.edu.tr) & Yalçın Cenik (yalcin.cenik@metu.edu.tr)

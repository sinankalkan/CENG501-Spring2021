# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper presents **Placepedia**, a comprehensive place dataset that contains images for places of interest from all over the world with massive attributes, where it is published in ECCV 2020. The authors of the paper developed **PlaceNet**, a unified framework for multi-level place recognition. It simultaneously predicts place item, category, function, city and country. In the image below, you can see the hierarchical structure of Placepedia with places from all over the world. 

![Placepedia](https://user-images.githubusercontent.com/53267971/127356409-e9169861-a7ca-4c34-9d3e-5233e668ef4b.PNG)


## 1.1. Paper summary

We can summarize the paper as below.
1) They build Placepedia, a large-scale place dataset with comprehensive annotations in multiple aspects. Compared to the other dataset related to place recognition tasks, Placepedia is the largest and the most comprehensive dataset for place understanding as you can see from the Table 1 below, which shows the comparision with the other dataset.
2) Authors designed four task-spesific benchmarks on Placepedia with respect to multi-faceted information.
3) They conduct systematic studies on place recognition, which demonstrate important challenges in place understanding as well as the connections between the visual characteristics and underlying socioeconomic or cultural implications.

In the related studies, the dataset focuses only the spesific tasks like place categorization, scene recognition, object retrieval or image retrieval. However, Placepedia can be used for all these task due to its much larger amount image and data, containing over 240K places with 35 million images labeled with 3K categories. Moreover, the function information of places may lead to a new task, namely place functio recognition.

**Table 1.** Comparing Placepedia with other existing datasets. Placepedia offers the
largest number of images and the richest information
![comparision](https://user-images.githubusercontent.com/53267971/127357466-9ba97ed1-2ead-4646-8c25-17514200bd3f.PNG)


# 2. The method and my interpretation

## 2.1. The original method

_**Datasets-Placepedia**_

– Places-Coarse. They select 200 places for validation and 400 places for testing,
from 50 famous cities of 34 countries. The remained 25K places are used
for training. For validation/testing set, we double checked the annotation
results. Places without category labels are manually annotated. After merging
similar items of labels, we obtain 50 categories and 10 functions. The training/validation/testing set have 5M/60K/120K images respectively, from 7K
cities of more than 200 countries.
– Places-Fine. Places-Fine shares the same validation/testing set with PlacesCoarse. For training set, they selected 400 places from the 50 cities of validation/testing places. Different from Places-Coarse, they also double checked
the annotation of training data. The training/validation/testing set have
110K/60K/120K images respectively, which are tagged with 50 categories, 10
functions, 50 cities, and 34 countries.

_**PlaceNet**_

**Loss Functions**. They study three losses for PlaceNet, namely, **softmax loss**, **focal
loss**, and **triplet loss**. 


**Pooling Methods**. They also study different pooling methods for PlaceNet,
namely, **average pooling**, **max pooling**, **spatial pyramid pooling**

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

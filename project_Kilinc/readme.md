
# Do We Need Zero Training Loss After Achieving Zero Training Error?

## 1. Introduction 

Zero training error is considered as important point to get general solution. Mostly, our neural networks is trained until it approaches to the zero loss point. This paper has an assumption that we do not need to wait unti zero loss. In fact learning until zero loss might be harmful. What we can do for obtainnig zero training error without obtaining zero loss is solved by a simple implementation. By one additional hyper-parameter which is flooding parameter, we can avoid approaching zero loss point while our training error is decreasing. 

Flooding parameter can effect the loss curve with the buoyancy. When the loss decreased below the flood level, backward operation starts to make gradient ascent to reach the flood level. This situation is considered as buoyancy effect. When the loss is above the flood level bacpropagation operation makes regular gradient descent method which is considered as gravity effect.  

![Flood Effect](flood.png)


## 2. Datasets 

We have two types of datasets which are sythetic and pytorch ready datasets, on the training part of the code you should indicate your dataset as given below, 

* Choose your input size and output size for "input_dim" and "output_dim" parameters. 

	* Sinusoidal dataset ::> input dim : 2 output dim : 1 class name : Sinusoidal_data
	* Two Gaussian dataset ::> input dim : 10 output dim : 1 class name : Gaussian_data
	* Spiral dataset ::> input dim : 2 output dim : 1 class name : Spiral_data
	* MNIST dataset ::> input dim : 28*28 output dim : 10 class name : MNIST_data
	* CIFAR10 dataset ::> input dim : 3072 output dim : 10 class name : Cifar10_data
	* CIFAR100 dataset ::> input dim : 3072 output dim : 100 class name : Cifar100_data
	* SVHN dataset ::> input dim : 32*32*3 output dim : 10 class name : SVHN_data
	* Kuzushiji dataset ::> input dim : 28*28 output dim : 10 class name : Kuzushiji_data
	
* Choose your nural network structure according to your needs. (Multi layer perceptron network is designed as similar with the neural network which is indicated in the paper) 

```ruby
model = Resnet_18(input_dim, class_number).ResNet18
model = MLP_Net(input_dim, class_number)
```

## 3. Implementation

Implementation of the method is simple. You can implement this metdhod easily for any training algorithm in pytorch as described below with the code block. By modifiying the loss parameter and backpropagating the modified loss parameter will make a flooding level effect on your loss curve. 


```ruby

outputs = model(inputs)
loss = criterion(outputs, labels)
flood = (loss-b).abs()+b # This is it!
optimizer.zero_grad()
flood.backward()
optimizer.step()

```

## 4. Results 

You can see the results from the paper below, 


The sythetic datasets are tested and the results are given below,

![Sythetic Results](synthetic.png)

Test results with pytorch's ready results are giveen below 

![Ready Datasets Results](ready.png)


You can see  an example result from my code, when we compare two results we can easily say that flood prevents training from overfitting problem. Red label indicates training loss, green label indicates the test loss. With flooding efeect we can train neural networks for more general solutions. 

* Gaussian data with flooding (flood_param = 0.17, noise = low ) 
![Flood Effect](gauss_with_flood.png)

* Gaussian data withot flooding (flood_param = 0.17, noise = low ) 
![Flood Effect](gauss_without_flood.png)

My resul table for sythetic dataset is given below, 

![My Accuracy table](my_res.png)


# References 


Ishida, T., Yamane, I., Sakai, T., Niu, G. &amp; Sugiyama, M.. (2020). Do We Need Zero Training Loss After Achieving Zero Training Error?. <i>Proceedings of the 37th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 119:4604-4614 Available from http://proceedings.mlr.press/v119/ishida20a.html.



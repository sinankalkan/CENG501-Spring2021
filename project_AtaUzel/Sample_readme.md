# Paper title

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction
The paper PARROT: Data-Driven Behavioral Priors for Reinforcement Learning, was presented in ICLR in 2021. The purpose of the algorithm that was introduced in the paper is to apply pre-training in reinforcement learning. In order to achieve that goal, they propose a method called behavioral priors. Pre-training is applied on different robotic manipulator tasks and it is observed that the pre-trained agents learn the task quicker compared to random agents.
Our aim is to implement a behvioral prior on a new environment and to compare the agent with the behavioral prior to an agent without behavioral prior.


## 1.1. Paper summary
In the existing literature, reinforcement learning agents go through a very long exploration period where almost no useful learning is achieved. However assuming the environment is composed of a robotic manipulator and objects, almost always, the manipulator must interact with the objects to achieve the task. As a prior, this narrows down the search space considerably. Therefore if such priors can be learned in order to bias the agents, it can be considered as a pre-training in reinforcement learning. This way, the learning process can be quicker.

# 2. The method and my interpretation

## 2.1. The original method

In the original method, the algorithm that is used in near optimal pick and place tasks are given briefly as "raise the manipulator", "go over the object", "lower the manipulator", "grab the object", "raise the object", "go over the target", "release the object". This is implemented for several environments including several objects. The actions taken according to the mentioned algorithm are saved along with an observation from a camera. This is repeated for different tasks.

For simulations, pybullet package is used in python. The object meshes and textures are used from [ShapeNet] (https://shapenet.org) dataset and pybullet objects.

Using the near optimal action-state pairs gathered from the simulations, a CNN called behavioral prior is trained end-to-end. As the input of the CNN, random gaussian noise and camera observations are used. As the output of the CNN, the near optimal actions are used.

After training the behavioral prior with the near optimal data, an agent which takes camera observations as inputs and gives 7D action vectors as outputs is used. The output of the agent is fed into the behavioral prior network and the output of the behavioral prior network is used as the final decision of the agent. This way, a random decision from the agent is biased into an action that could be useful in other tasks. Then, a suitable reinforcement learning algorithm, SAC in this case, can be used in order to train the agent to control the environment through the behavioral prior network. SAC is suitable for the expermients given in the paper since it allows reinforcement learning in continuos observation and continuous action spaces.

The output of the behavioral prior is used as a combination of three vector: the position of the end effector, the orientation of the end effector, and the grip action. Joint angles are calculated using inverse kinematics. The value reward function in reinforcement learning is 1 if the task if successfuly complete, 0 if it is not.

The environment in the reinforcement learning is composed of 3 three objects and the manipulator. The agents task is to either pick a specific object and raise it or pick a specific object and place it on another specific object.

## 2.2. My interpretation 

In the paper, the algorithm that was used in order to get near optimal action-state pairs was very brief. Most  of the implementation details were chosen properly. As examples, the maximum velocity and torque that can be achieved by the robotic manipulator joints, how much the end effector should be raised before it moves over an object, how fast the object is carried, what the orientation of the end effector is while carrying the objects can be given.

Also, the robotic arm model and the objects selected from ShapeNet were not given in the paper. A proper robotic arm model and proper objects from ShapeNet dataset was chosen properly. Unlike the original paper, no objects from pybullet package were used in our implementation. In the simulation, the objects were scaled properly and pyhsical properties are arbitrarily given to the objects. The camera

In the original paper, the learning method of the behavioral prior was not given. As an interpretation, we decided to take random batches from a relatively big dataset and train on them. This way, the number of epochs was not chosen as a parameter.

In reinforcement learning, we used a different reward function in order to make learning easier. The reward function was chosen such that it rewards the agent when the manipulator gets closer to the target object. Also, the reward was increased as the object was raised.

In our interpretation, the tasks are limited to picking a specific object from a similar environment with the original paper.

# 3. Experiments and results

## 3.1. Experimental setup

In order to evaluate the method, two agents were created where one of them had a behavioral prior and one of them did not. After using the reinforcement learning algorithm (SAC), on the agents, their success over time is compared. Success is evaluated by the mean reward over different environment episodes. Different from the original paper, in our implementation, mean reward is considered instead of success rate.

## 3.2. Running the code

```
parrot
│   behavioral_prior.py
│   q_learning.py
│   robot_arm.py
│
│─── bp models
│
│─── data3
│
│─── evals
│
│─── sample models
│
│─── reactor_description
│
│─── selected textures
```

In order to repeat the experiments, one should firstly run "robot_arm.py" file in order to generate near optimal action-state pairs in "data3" folder. When there is enough data, one can exit the script.
Then, q_learning.py should be run in order to train a behavioral prior network based on the generated near optimal action-state pairs. This script will save the trained models inside "bp models" folder.
Finally, one should run "behavioral_prior.py" script. This script will use the final behavioral prior model and start reinforcement learning.

## 3.3. Results

Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

Discuss the paper in relation to the results in the paper and your results.

# 5. References

[PARROT: Data-Driven Behavioral Priors for Reinforcement Learning] (https://arxiv.org/abs/2011.10024)
[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor] (https://arxiv.org/abs/1801.01290)
[Robotic Manipulator] (https://github.com/a-price/reactor_description)


# Contact

Denge Uzel - uzel.denge@metu.edu.tr
Erdem Ata - erdem.ata@metu.edu.tr

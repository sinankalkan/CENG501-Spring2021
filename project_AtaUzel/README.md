# PARROT: DATA-DRIVEN BEHAVIORAL PRIORS FOR REINFORCEMENT LEARNING

This readme file is an outcome of the [CENG501 (Spring 2021)](http://kovan.ceng.metu.edu.tr/~sinan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2021) Project List](https://github.com/sinankalkan/CENG501-Spring2021) for a complete list of all paper reproduction projects.

# 1. Introduction
The paper PARROT: Data-Driven Behavioral Priors for Reinforcement Learning, was presented in ICLR in 2021. The purpose of the algorithm that was introduced in the paper is to apply pre-training in reinforcement learning. In order to achieve that goal, they propose a method called behavioral priors. Pre-training is applied on different robotic manipulator tasks and it is observed that the pre-trained agents learn the task quicker compared to random agents by having a better initial performance.
Our aim is to implement a behvioral prior on a new environment and to compare the agent with the behavioral prior to an agent without behavioral prior.


## 1.1. Paper summary

In the existing literature, reinforcement learning agents go through a very long exploration period where almost no useful learning is achieved. However assuming the environment is composed of a robotic manipulator and objects, almost always, the manipulator must interact with the objects to achieve the task. As a prior, this narrows down the search space considerably. Therefore if such priors can be learned in order to bias the agents, it can be considered as a pre-training in reinforcement learning. This way, the learning process can be quicker.

In order to achieve pre-training, the action-state pairs that are obtained from similar tasks are learned by a network called behavioral prior network. The behavioral prior network's structure is a conditional real NVP with four affine coupling layers. At the decision making process, this network takes the output of the agent and provides an action to the environment. The behavioral prior network can be considered as a tool for modifying the random output of the agent into an action which is more likely to result in a high reward. After the behavioral prior network is trained, its weights can be freezed.

Finally, a deep learning agent can be trained using a new task.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126991041-e1c35042-1c1b-492f-946a-579b617a77b7.png"/>
</p>
<p align="center">
  Figure 1: Trajectories of the manipulator with and without behavioral priors using a random policy given in the original paper
</p>


# 2. The method and our interpretation

## 2.1. The original method

In the original method, the algorithm that is used in near optimal pick and place tasks are given briefly as "raise the manipulator", "go over the object", "lower the manipulator", "grab the object", "raise the object", "go over the target", "release the object". This is implemented for several environments including several objects. The actions taken according to the mentioned algorithm are saved along with an observation from a camera. This is repeated for different tasks.

For simulations, [PyBullet](pybullet.org) package is used in python. The object meshes and textures are used from [ShapeNet](https://shapenet.org) dataset and PyBullet objects.

Using the near optimal action-state pairs gathered from the simulations, a CNN called behavioral prior is trained end-to-end. As the input of the CNN, random gaussian noise whose mean is 0 and standard deviation is 1 and camera observations (48x48x3) are used. As the output of the CNN, the near optimal actions are used.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126990300-8c1a8a3b-e8d0-4f15-8a0a-c313f3d30624.png"/>
</p>
<p align="center">
  Figure 2: The structure of the behavioral prior network from the original paper
</p>

After training the behavioral prior with the near optimal data, an agent which takes camera observations as inputs and gives 7D action vectors as outputs is used. The output of the agent is fed into the behavioral prior network and the output of the behavioral prior network is used as the final decision of the agent. This way, a random decision from the agent is biased into an action that could be useful in other tasks. Then, a suitable reinforcement learning algorithm, SAC in this case, can be used in order to train the agent to control the environment through the behavioral prior network. SAC is suitable for the expermients given in the paper since it allows reinforcement learning in continuos observation and continuous action spaces.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126990430-0e188b3a-0646-41a5-8393-33c9c5319119.png"/>
</p>
<p align="center">
  Figure 3: The structure of the policy network from the original paper
</p>


The output of the behavioral prior is used as a concatenation of three vectors: the position of the end effector, the orientation of the end effector, and the grip action. Joint angles are calculated using inverse kinematics. The value reward function in reinforcement learning is 1 if the task if successfuly complete, 0 if it is not.

The environment in the reinforcement learning is composed of 3 three objects and the manipulator. The agent's task is to either pick a specific object and raise it or pick a specific object and place it on another specific object.


<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126990755-0558aea2-8458-42e4-89d1-a521f792684a.png"/>
</p>
<p align="center">
  Figure 4: Problem setting from the original paper
</p>


## 2.2. Our interpretation 

In the paper, the algorithm that was used in order to get near optimal action-state pairs was very brief. Most  of the implementation details were chosen properly. As examples, the maximum velocity and torque that can be achieved by the robotic manipulator joints, how much the end effector should be raised before it moves over an object, how fast the object is carried, what the orientation of the end effector is while carrying the objects can be given.

Also, the robotic arm model and the objects selected from ShapeNet were not given in the paper. A proper robotic arm model and proper objects from ShapeNet dataset was chosen. Similar to the original paper, the robotic arm model has 6 degrees of freedom and a 2 finger gripper. Unlike the original paper, no objects from pybullet package were used in our implementation. In the simulation, the objects were scaled properly and pyhsical properties are arbitrarily given to the objects. The camera position and orientation was also chosen similar to the original paper.

In the original paper, the learning method of the behavioral prior was not given. As an interpretation, we decided to take random batches from a relatively big dataset composed of 250k action-state pairs and train on them. This way, the number of epochs was not chosen as a parameter. The loss function that was used in the training of the behavioral prior was also not given. As a suitable loss function, mean square error function was used. In the training, a learning rate of 0.001 was used with a batch size of 256. Traning was stopped after 100 batches. The input to the network was a 48x48x3 observation image and a sample from 8D gaussian noise whose mean is 0 and standard deviation is 1. As an output the 8D vector that represents the position and orientation of the end effector and the grip action parameters is used.

Unlike the original paper, the evalution task was chosen as a relatively simpler task. The agent was rewarded when the manipulator contacted target objects in the environment. In the original paper, evaluation tasks included pick tasks and place tasks.

- Target network update period: 100
- Discount factor 0.99
- Learning rate for both actor and critic: 0.001
- Reward scale: 1
- Gradient steps per environment step: 1

In our interpretation, the tasks are limited to picking a specific object from a similar environment with the original paper.

<p align="center">
  <video src="https://user-images.githubusercontent.com/61411406/126642521-d8a0d4b0-6dc0-46a6-b1e6-7e5d20b32910.mp4"/>
</p>
<p align="center">
  The near-optimal pick and place policy
</p>


# 3. Experiments and results

## 3.1. Experimental setup

In order to evaluate the method, two agents were created where one of them had a behavioral prior and one of them did not. After using the reinforcement learning algorithm (SAC), on the agents, their success over time is compared. Success is evaluated by the mean reward over different environment episodes. Two different experiments were used in order to evaluate the method.

In both experiments, the specific task that was chosen to evaluate the method requires the agent to contact on of the two chosen objects. The environment includes three objects. If the robotic manipulator contacts the third object, no reward is given but the simulation continues.

### 3.1.1. Experiment 1

Since the time and resources were limited for the experiments, the first experiment was focused on only the beginning of the reinforcement learning process. However the reinforcement learning algorithm was run for 8 seeds which is higher compared to the other experiment.

The behavioral prior is expected to result in a better average reward at the beginning of the learning. However, in this experiment, since the reinforcement learning algorithm was stopped before the agent completely solved the problem, it is not known whether the task would be solved sooner for the agent with the behavioral prior.

This task is learned for 5k timesteps and an evaluation was made every 50 timesteps. In order to eliminate the effect of the object locations, learning algorithm was run on 8 different random seeds where the positions of three objects vary.

### 3.1.2. Experiment 2

In this experiment, the reinforcement learning algorithm was run for 50k timesteps which is higher compared to the first experiment. Similar to the first experiment, an evaluation was made every 50 timesteps. Since the algorithm was run for longer, the average rewards significantly increased during the learning unlike the other experiment. However due to limitations in time and resources, this experiment was applied on 3 seeds which is lower than the first experiment.

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
│    │─── with
│    └─── without
│
│─── sample models
│
│─── reactor_description
│
└─── selected textures
```

In order to repeat our experiments, one should firstly run "robot_arm.py" file in order to generate near optimal action-state pairs in "data3" folder. When there is enough data, one can exit the script.

Then, "behavioral_prior.py" should be run in order to train a behavioral prior network based on the generated near optimal action-state pairs. This script will save the trained models inside "bp models" folder.

Finally, one should run "q_learning.py" script. This script will use the final behavioral prior model and start reinforcement learning. Then store the results in "evals" folder. "With" and "without" foders inside the "evals" folder are used in order to keep the logs of the reinforcement learning agents with and without behavioral prior.  

## 3.3. Results

### 3.3.1. Results of Behavioral Prior Training

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126654521-3922a2a2-e0fc-46e9-959c-46209b436675.png" />
</p>

<p align="center">
  Figure 9: Loss of the behavioral prior network vs. Iteration Number
</p>

In the figure given above, the change of the loss of the behavioral prior network over the number of learning iterations is given. It looks like the training of the behavioral prior network was successful and not a lot of overfitting occured.

### 3.3.1. Results of Experiment 1

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126568014-5dd670ef-6733-47f8-93a3-3dccb85b6e31.png" />
</p>

<p align="center">
  Figure 5: Average Reward vs. Number of evaluations without Prior for Experiment 1
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126568029-3c114066-3479-402c-9452-bb295c45e00a.png" />
</p>

<p align="center">
  Figure 6: Average Reward vs. Number of evaluations with Prior for Experiment 1
</p>

The experiments are not exactly the same with the original paper. The experiments that was used in our implementation is explained in the previous section. The figures given above show the success rate (average reward) of the agent at the beginning of the reinforcement learning in experiment 1.

The average of the success rates given in the Figure 5 is 21.5% and the average of the success rates given in the Figure 6 is 1.75%. As expected, the success rate of the agent with behavioral prior is significantly higher compared to the agent without behavioral prior at the beginning of the reinforcement learning.

### 3.3.2. Results of Experiment 2

<p align="center">
  <img src="" />
</p>

<p align="center">
  Figure 7: Average Reward vs. Number of evaluations without Prior for Experiment 2
</p>

<p align="center">
  <img src="" />
</p>

<p align="center">
  Figure 8: Average Reward vs. Number of evaluations with Prior for Experiment 2
</p>

The average of the success rates given in the Figure 7 is *todo* and the average of the success rates given in the Figure 8 is *todo*. As expected, the success rate of the agent with behavioral prior gets significantly higher with much less learning compared to the agent without behavioral prior. The plots are prepared using a running average of window size *todo* to make them easily readable.

Since the number of seeds that was used in this experiment is relatively low, the results may not be very generalizable.

### 3.3.3. Some Demonstrations of the Results

<p align="center">
  <video src="https://user-images.githubusercontent.com/61411406/126642798-0c9205f0-8f96-4303-9f8a-026eddb9e586.mp4"/>
</p>
<p align="center">
  A test environment where behavioral prior is useful
</p>

<p align="center">
  <video src="https://user-images.githubusercontent.com/61411406/126643118-41c01f46-a128-4393-ac40-7e8be4e6aa34.mp4"/>
</p>
<p align="center">
  The same test environment where the agent does not use behavioral prior
</p>

# 4. Conclusion

The results in the original paper are consisntent with our results. In both cases, it was observed that the agent with the behavioral prior starts the reinforcement learning with an obvious advantage.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61411406/126569511-3b6306eb-a92b-41ad-a486-93dce0c79756.png" />
</p>

<p align="center">
  Figure 10: Comparison of methods from the original paper
</p>


As the figure suggests, the method proposed in the paper starts with a high average reward compared to other methods. This result was confirmed with our first experiment. The method is also expected to learn to task more rapidly compared to other methods. This result was confirmed with our second experiment. The results were discussed in the previous section.

# 5. References

[1] [Singh, A., Liu, H., Zhou, G., Yu, A., Rhinehart, N., &amp; Levine, S. (2020, November 19). Parrot: Data-driven behavioral priors for reinforcement learning.](https://arxiv.org/abs/2011.10024)

[2] [Haarnoja, T., Zhou, A., Abbeel, P., &amp; Levine, S. (2018, August 8). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.](https://arxiv.org/abs/1801.01290)

[3] [A-Price. A-Price/Reactor_Description: URDF model for Phantomx REACTOR robot arm. ](https://github.com/a-price/reactor_description)

[4] [Dinh, L., Sohl-Dickstein, J., &amp; Bengio, S. (2017, February 27). Density estimation using real nvp.](https://arxiv.org/abs/1605.08803)


# Contact

Denge Uzel - uzel.denge@metu.edu.tr

Erdem Ata - erdem.ata@metu.edu.tr

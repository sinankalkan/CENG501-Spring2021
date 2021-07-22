# -*- coding: utf-8 -*-
"""

CENG501 - Spring 2021 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

from behavioral_prior import I2P, NVP, BP
from stable_baselines3.common.env_checker import check_env
from robot_arm import Robot
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import SAC
import stable_baselines3
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from gym import spaces

BOX_DIR = "./sample models/box"
OTHER_DIR = "./sample models/other"
TEXT_DIR = "./selected textures"
BOX_SIZE = 0.06
OTHER_SIZE = 0.03
MAX_DIST_TO_ARM = 0.25
MIN_DIST_TO_ARM = 0.12
MIN_DIST_OBJ = 0.1
MIN_ANGLE = 3*np.pi/2 + np.pi/4
MAX_ANGLE = 3*np.pi/2 + 3*np.pi/4
SCALE_FACTOR = np.array([0.8, 0.8, 1])
DATA_PATH = "./data3"

MOVING_OBJECT_INDEX = 0
# TARGET_OBJECT_INDEX = 1
TARGET_OBJECT_INDEX = 2

SUCCESS_REWARD = 1
FAIL_REWARD = 0
MAX_STEPS = 100

OBS_WIDTH = 48
OBS_HEIGHT = 48
LOG_PATH_WITH = "./evals/with"
LOG_PATH_WITHOUT = "./evals/without"

path1 = f"{OTHER_DIR}/4ddc9ff12f94fd4f3c143af07c12991a.obj"
path2 = f"{OTHER_DIR}/5bbc259497af2fa15db77ed1f5c8b93.obj"
path3 = f"{BOX_DIR}/e500097e7023db9ac951cf8670bfff6.obj"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(PolicyCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 1, padding = 1),
            nn.MaxPool2d(4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride = 1, padding = 1),
            nn.MaxPool2d(4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)

policy_kwargs = dict(
    features_extractor_class=PolicyCNN,
    features_extractor_kwargs=dict(features_dim=2304),
    net_arch=[1024, 512, 256]
)




for i in range(10, 18):
    seed = i

    env = Robot("reactor_description/robots/reactor_description.URDF", path1, path2, path3, seed, gui = False, prior = True)
    eval_env = Robot("reactor_description/robots/reactor_description.URDF", path1, path2, path3, seed, gui = False, prior = True)

    model = SAC("CnnPolicy", env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs, gamma = 0.99, learning_rate = 3e-3, ent_coef = 1, target_update_interval = 100, gradient_steps = 1, learning_starts = 10)
    model.learn(total_timesteps=5000, log_interval=4, eval_env = eval_env, eval_freq = 50, n_eval_episodes = 1, eval_log_path = f'{LOG_PATH_WITH}/{i}')
    
    env.close()
    
for i in range(10, 18):
    seed = i

    env = Robot("reactor_description/robots/reactor_description.URDF", path1, path2, path3, seed, gui = False, prior = False)
    eval_env = Robot("reactor_description/robots/reactor_description.URDF", path1, path2, path3, seed, gui = False, prior = False)

    model = SAC("CnnPolicy", env, verbose=1, buffer_size=10000, policy_kwargs=policy_kwargs, gamma = 0.99, learning_rate = 3e-3, ent_coef = 1, target_update_interval = 100, gradient_steps = 1, learning_starts = 10)
    model.learn(total_timesteps=5000, log_interval=4, eval_env = eval_env, eval_freq = 50, n_eval_episodes = 1, eval_log_path = f'{LOG_PATH_WITHOUT}/{i}')
    
    env.close()

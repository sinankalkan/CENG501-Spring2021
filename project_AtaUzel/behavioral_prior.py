# -*- coding: utf-8 -*-
"""

CENG501 - Spring 2021 

"""

import torch
import torchvision
import cv2
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

IMAGE_SHAPE = [48,48,3]
BATCH_SIZE = 256
EPOCHS = 1000
DATA_PATH = "./data3"

data_files = os.listdir(DATA_PATH)
data_names = {name.split('.')[0] for name in data_files}
data_names = np.array(list(data_names))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_random_name_batch(data_names, batch_size):
    return np.random.choice(data_names, batch_size)

def load_batch(names):
    images = []
    actions = []
    for n in names:
        image = cv2.imread(os.path.join(DATA_PATH, f"{n}.png"))/255
        image = np.einsum('hwc->chw', image)
        images.append(image)
        action = np.load(os.path.join(DATA_PATH, f"{n}.npy"))
        action = np.append(action, action[-1])
        actions.append(action)

    images = np.array(images)
    actions = np.array(actions)
    return images, actions


# names = get_random_name_batch(data_names, 3)
# images, actions = load_batch(names)


class I2P(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class NVP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(260, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 8)

    def forward(self, z, phi):
        # z, phi = d
        if type(z) == np.ndarray:
            z = torch.from_numpy(z)
        z1, z2 = z[:, :4], z[:, 4:]
        x = torch.cat((z1, phi), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        v = x[:, :4]
        t = x[:, 4:]
        
        z_p = torch.empty(z.shape[0], 8)
        z_p[:, :4] = z1
        z_p[:, 4:] = z2*torch.exp(v) + t
        
        return z_p


class BP(nn.Module):
    def __init__(self):
        super().__init__()
        self.nvp = NVP()
        self.i2p = I2P()

    def forward(self, z, img):
        # z, img = d
        phi = self.i2p(torch.from_numpy(img))
        z_p = self.nvp(z, phi)
        z_p = self.nvp(z_p, phi)
        z_p = self.nvp(z_p, phi)

        return z_p
    
    def parameters(self):
        return list(self.nvp.parameters()) + list(self.i2p.parameters())

if __name__ == '__main__':
    i2p = I2P()
    nvp = NVP()
    bp = BP()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(bp.parameters(), lr=0.001, momentum=0.9)


    running_loss = 0.0
    for i in range(EPOCHS):
        try:
            names = get_random_name_batch(data_names, BATCH_SIZE)
            images, actions = load_batch(names)
            for a in actions:
                print(a)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = images.astype(np.float32), torch.from_numpy(actions.astype(np.float32))
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            z = np.random.normal(size=(BATCH_SIZE, 8)).astype(np.float32)

            outputs = bp(z, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (i, i + 1, running_loss))
            if i%100 == 0:
                torch.save(bp, f"./bp models/{str(time.time())[:10]}.pt")
            running_loss = 0.0
        except Exception as e:
            print(e)   

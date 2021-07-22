# -*- coding: utf-8 -*-
"""

CENG501 - Spring 2021 

"""

import pybullet
import pybullet_data
import numpy as np
import time
import os
import random
from scaling import get_dimensions
import cv2
import gym
from gym import spaces
import torch
from behavioral_prior import I2P, NVP, BP

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
BP_FOLDER = "./bp models"
BP_PATH = f"./bp models/{sorted(os.listdir(BP_FOLDER))[-1]}"

MOVING_OBJECT_INDEX = 0
# TARGET_OBJECT_INDEX = 1
TARGET_OBJECT_INDEX = 2

SUCCESS_REWARD = 1
FAIL_REWARD = 0
MAX_STEPS = 100

OBS_WIDTH = 48
OBS_HEIGHT = 48

LOAD_BP = True

SAVE = False # Data will be saved if this is True

class Robot(gym.Env):
    def __init__(self, urdf_path, path1 = None, path2 = None, path3 = None, seed = None, gui = True, prior = True):
        super(Robot, self).__init__()
        if gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        self.urdf_path = urdf_path
        self.joint_ranges = np.array([np.pi*2, np.pi*1.5, np.pi*1.5, 4, np.pi*2, 0.014, 0.014])
        self.num_steps = 0
        self.observation_space = spaces.Box(0, 255, [OBS_HEIGHT, OBS_WIDTH, 3], dtype = np.uint8)
        self.action_space = spaces.Box(np.array([-3, -3, -3, -3, -3, -3, -3,-3])/2,\
                                       np.array([3, 3, 3, 3, 3, 3, 3, 3])/2)
        self.gripper_state = 0
        if LOAD_BP:
            self.bp = torch.load(BP_PATH)
            
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.seed = seed
        self.prior = prior


    
    def move_arm_to(self, position, orientation = None): 
        if orientation is None:
            joint_positions = pybullet.calculateInverseKinematics(bodyUniqueId = self.id,\
                                    endEffectorLinkIndex = self.EE_LINK, targetPosition = position)
        else:
            joint_positions = pybullet.calculateInverseKinematics(bodyUniqueId = self.id,\
                                    endEffectorLinkIndex = self.EE_LINK, targetPosition = position, targetOrientation = orientation)
        
        self.set_joint_state(joint_positions[:5], [0,1,2,3,4], [1]*5)
        
    def get_joint_state(self):
        return pybullet.getJointStates(self.id, range(self.num_of_joints))
    
    def set_joint_state(self, joint_positions, joints, forces):
        for jp, j, f in zip(joint_positions, joints, forces):
            self.target[j] = jp
            pybullet.setJointMotorControl2(self.id, j, pybullet.POSITION_CONTROL,\
                                            jp, force = f, maxVelocity = 3)
        
    def target_reached(self):
        return np.linalg.norm(np.array(self.get_joint_state())[:5,0] - np.array(self.target[:5])) < 2*1e-2
    
    def grasp(self, close = 1):
        self.gripper_state = close
        self.set_joint_state([0.014*close, 0.014*close], [5,6], [0.15,0.15])
    
    def get_state(self):
        view_matrix = pybullet.computeViewMatrix([0.45,0,0.25], [0.2,0,0.1], [-np.sin(self.view_angle),0,np.cos(self.view_angle)])
        proj_matrix = pybullet.computeProjectionMatrixFOV(50,1,0.1,100)
        
        return pybullet.getCameraImage(48,48, view_matrix, proj_matrix)[2][:,:,:3]
    
    def get_reward(self):
        cps = pybullet.getContactPoints()
        reward = 0
        for cp in cps:
            contact_ids = [cp[1], cp[2]]
            if self.bodies[0] in contact_ids and self.id in contact_ids:
                reward = 1
            if self.bodies[1] in contact_ids and self.id in contact_ids:
                reward = 1
            if self.bodies[2] in contact_ids and self.id in contact_ids:
                reward = 0

        return reward
    
    def update(self, n = 10, sleep = 0):
        self.save_data()
        state = self.get_state()
        for _ in range(n):
            pybullet.stepSimulation()
            self.num_steps += 1
        next_state = self.get_state()
        if self.num_steps > MAX_STEPS:
            self.done = True
        
    
    def step(self, action):

        if self.prior:
            z = np.array([action])
            z = torch.from_numpy(z.astype(np.float32))
            image = np.einsum('hwc->chw', self.get_state())
            image = np.array([image]).astype(np.float32)
    
            a = self.bp(z, image)
        else:
            a = np.array([action])
        pos=a[0,:3]
        euler = a[0,3:6]
        quaternion = pybullet.getQuaternionFromEuler(euler)
        self.move_arm_to(pos, quaternion)
        grasp = 1 if a[0,6] + a[0,7] > 1 else 0
               
        self.grasp(grasp)
        self.update(3)
        reward = self.get_reward()
        if reward == 1:
            self.done = True
        return (self.get_state(), reward, self.done, {})
        
    def render(self):
        return self.get_state()

    def close(self):
        pybullet.disconnect()
        
    def pick(self, obj_):
        euler_orientation = [1.57,1.57,1.57]
        orientation = pybullet.getQuaternionFromEuler(euler_orientation)
        
        o_pos, o_orien = pybullet.getBasePositionAndOrientation(obj_)
        
        first_target = o_pos + np.array([0,0,0.2])
        second_target = o_pos + np.array([0,0,0.06])
        
        sleep = 0
        self.move_arm_to(first_target, orientation)
        while not self.target_reached():
            self.move_arm_to(first_target, orientation)
            self.update(10, sleep)
        
        self.grasp(0)
        self.update(100, sleep)
        
        num_of_targets = 10
        for i in range(num_of_targets):
            target = first_target-(first_target-second_target)*i/(num_of_targets-1)
            self.move_arm_to(o_pos + np.array(target), orientation)
            while not self.target_reached():
                self.move_arm_to(np.array(target), orientation)
                self.update(10, sleep)
        
        self.grasp(1)
        self.update(100, sleep)
        
        self.move_arm_to(first_target)
        while not self.target_reached():
            self.move_arm_to(first_target, orientation)
            self.update(10, sleep)
        
        
    def place(self, obj_):
        euler_orientation = [1.57,1.57,1.57]
        orientation = pybullet.getQuaternionFromEuler(euler_orientation)
        
        o_pos, o_orien = pybullet.getBasePositionAndOrientation(obj_)
        
        target_pos = np.array(o_pos)
        target_pos[2] = 0.2
        
        sleep = 0
        self.move_arm_to(target_pos, orientation)
        while not self.target_reached():
            self.move_arm_to(target_pos, orientation)
            self.update(10, sleep)
        
        self.grasp(0)
        self.update(500, sleep)
    
    def save_data(self):
        if not SAVE:
            return
        file_name = str(time.time()).replace('.', '')
        view_matrix = pybullet.computeViewMatrix([0.45,0,0.25], [0.2,0,0.1], [-np.sin(self.view_angle),0,np.cos(self.view_angle)])
        proj_matrix = pybullet.computeProjectionMatrixFOV(50,1,0.1,100)
        
        img = pybullet.getCameraImage(48,48, view_matrix, proj_matrix)[2]
        pos, orientation, _, _, _, _ = pybullet.getLinkStates(self.id, [self.EE_LINK])[0]
        euler = pybullet.getEulerFromQuaternion(orientation)
        pos = np.array(pos)
        euler = np.array(euler)
        target_xyz = np.append(pos, euler)
        if type(self.gripper_state) == int:
            gs = self.gripper_state
        else:
            gs = self.gripper_state.detach().numpy()
        target_xyz = np.append(target_xyz, gs)
        cv2.imwrite(f"{DATA_PATH}/{file_name}.png", img)
        np.save(f"{DATA_PATH}/{file_name}", target_xyz)
    
    def reset(self):
        print("Reset")
        pybullet.resetSimulation()
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setTimeStep(0.01)
        self.num_steps = 0
        self.plane = pybullet.loadURDF("plane.urdf", useFixedBase=1)
        texUid = pybullet.loadTexture("textures/fe690c8553d7a098.jpg")
        pybullet.changeVisualShape(self.plane, -1, textureUniqueId=texUid)
        self.id = pybullet.loadURDF(self.urdf_path, useFixedBase=1, basePosition = [0,0,0])
        self.num_of_joints = pybullet.getNumJoints(self.id)
        self.view_angle = 10*np.pi/180

        
        self.joint_states = self.get_joint_state()
        self.EE_LINK = 4
        self.GRIPPER_LINKS = [5, 6]
        self.target = np.array([0 for i in range(self.num_of_joints)], dtype = "float")
        pybullet.setGravity(0, 0, -9.81)   # everything should fall down
        pybullet.setRealTimeSimulation(0)  # we want to be faster than real time :)
        self.bodies = create_scene(self.path1, self.path2, self.path3, self.seed)
        self.init_target_pos, _ = pybullet.getBasePositionAndOrientation(self.bodies[TARGET_OBJECT_INDEX])
        self.init_target_pos = np.array(self.init_target_pos)
        self.done = False
        return self.get_state()
        
    def get_data_loop(self):
        for i in range(50_000):
            pybullet.resetSimulation()
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet.setTimeStep(0.01)
            self.num_steps = 0
            self.plane = pybullet.loadURDF("plane.urdf", useFixedBase=1)
            texUid = pybullet.loadTexture("textures/fe690c8553d7a098.jpg")
            pybullet.changeVisualShape(self.plane, -1, textureUniqueId=texUid)
            self.id = pybullet.loadURDF(self.urdf_path, useFixedBase=1, basePosition = [0,0,0])
            self.num_of_joints = pybullet.getNumJoints(self.id)
            self.view_angle = 10*np.pi/180

            
            self.joint_states = self.get_joint_state()
            self.EE_LINK = 4
            self.GRIPPER_LINKS = [5, 6]
            self.target = np.array([0 for i in range(self.num_of_joints)], dtype = "float")
            pybullet.setGravity(0, 0, -9.81)   # everything should fall down
            pybullet.setRealTimeSimulation(0)  # we want to be faster than real time :)
            
            self.bodies = create_scene(self.path1, self.path2, self.path3, self.seed)
            self.init_target_pos, _ = pybullet.getBasePositionAndOrientation(self.bodies[TARGET_OBJECT_INDEX])
            self.init_target_pos = np.array(self.init_target_pos)
            self.pick(self.bodies[MOVING_OBJECT_INDEX])
            self.place(self.bodies[TARGET_OBJECT_INDEX])
            
            print(i+1)

def create_scene(path1 = None, path2 = None, path3 = None, seed = None):
    p = [None, None, None]
    p[0] = path1 or get_random(OTHER_DIR)
    p[1] = path2 or get_random(OTHER_DIR)
    p[2] = path3 or get_random(BOX_DIR)
    centers = []
    sizes = []
    scales = [None, None, None]
    for i in range(3):
        c, s = get_dimensions(p[i])
        centers.append(c)
        sizes.append(s)
    if not seed is None:
        np.random.seed(seed)
    scales[0] = SCALE_FACTOR * OTHER_SIZE/sizes[0]
    scales[1] = SCALE_FACTOR * OTHER_SIZE/sizes[1]
    scales[2] = BOX_SIZE/sizes[2]
    positions = []
    random_pos = np.random.random(size=(2)) * np.array([MAX_ANGLE - MIN_ANGLE, MAX_DIST_TO_ARM - MIN_DIST_TO_ARM]) + np.array([MIN_ANGLE, MIN_DIST_TO_ARM])
    rect_coord = np.array([np.cos(random_pos[0])*random_pos[1], np.sin(random_pos[0])*random_pos[1]])
    positions.append(rect_coord)
    for i in range(2):
        random_pos = np.random.random(size=(2)) * np.array([MAX_ANGLE - MIN_ANGLE, MAX_DIST_TO_ARM - MIN_DIST_TO_ARM]) + np.array([MIN_ANGLE, MIN_DIST_TO_ARM])
        rect_coord = np.array([np.cos(random_pos[0])*random_pos[1], np.sin(random_pos[0])*random_pos[1]])
        while get_min_dist(rect_coord, positions) < MIN_DIST_OBJ:
            random_pos = np.random.random(size=(2)) * np.array([MAX_ANGLE - MIN_ANGLE, MAX_DIST_TO_ARM - MIN_DIST_TO_ARM]) + np.array([MIN_ANGLE, MIN_DIST_TO_ARM])
            rect_coord = np.array([np.cos(random_pos[0])*random_pos[1], np.sin(random_pos[0])*random_pos[1]])
        
        positions.append(rect_coord)
    
    for i in range(len(positions)):
      positions[i] = [positions[i][0], positions[i][1], 0.03]  
    
    ids = []
    for i in range(3):
        ids.append(load_object(p[i], scales[i], centers[i], positions[i]))
    np.random.seed(int(np.random.random()*100))
    return ids

def get_number(name, dir_):
    model_names = os.listdir(dir_)
    try:
        return model_names.index(name)
    except:
        return -1

def get_min_dist(p, cs):
    dists = []
    for c in cs:
        dists.append(np.linalg.norm(c-p))
    return min(dists)

def get_random(dir_):
    model_names = os.listdir(dir_)
    model_names = [m for m in model_names if m.split('.')[1] == "obj"]
    random_name = model_names[int(random.random()*len(model_names))]
    path = f"{dir_}/{random_name}"
    return path

def load_object(path, scale, center, position):
    shape_id = pybullet.createVisualShape(pybullet.GEOM_MESH, fileName = path, meshScale = scale, visualFramePosition=-center*scale)
    coll_id = pybullet.createCollisionShape(pybullet.GEOM_MESH, fileName = path, meshScale = scale, collisionFramePosition=-center*scale)
    euler_orientation = [0,0,0]
    orientation = pybullet.getQuaternionFromEuler(euler_orientation)
    bodyUid = pybullet.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0,0,0],
                                baseVisualShapeIndex=shape_id,
                                basePosition=position,
                                baseCollisionShapeIndex=coll_id,
                                baseInertialFrameOrientation=orientation)
    
    name = path.split('/')[-1]
    texture_number = get_number(name, OTHER_DIR)
    others = os.listdir(OTHER_DIR)
    if texture_number == -1:
        texture_number = get_number(name, BOX_DIR) + len(others)
    
    textures = os.listdir(TEXT_DIR)
    texture = textures[texture_number]
    texUid = pybullet.loadTexture(f"{TEXT_DIR}/{texture}")
    pybullet.changeVisualShape(bodyUid, -1, textureUniqueId=texUid)
    pybullet.changeDynamics(bodyUid, -1, 0.01, 5, 0.1, 0.1, 0.1)
    
    return bodyUid


if __name__ == "__main__":
    LOAD_BP = False
    r = Robot("reactor_description/robots/reactor_description.URDF")
    r.get_data_loop()

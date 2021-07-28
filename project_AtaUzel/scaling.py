# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:52:05 2021

@author: erdemata
"""

import random
import os
import numpy as np



def get_dimensions(path):
    f = open(path)
    
    lines = f.readlines()
    
    lines = [l.strip().split(' ')[1:] for l in lines if l.split(' ')[0] == 'v']
    
    for i in range(len(lines)):
        for j in range(3):
            lines[i][j] = float(lines[i][j])
    
    faces = np.array(lines)
    
    centers = np.array([0.,0.,0.])
    sizes = np.array([0.,0.,0.])
    
    for i in range(3):
        is_ = faces[:,i]
        # i_center = (np.min(is_) + np.max(is_))/2
        centers[i] = np.mean(is_)
        sizes[i] = (np.max(is_) - np.min(is_))
    
    return centers, sizes

def arr2str(arr):
    res = [str(a) for a in arr]
    return " ".join(res)



if __name__ == "__main__":
    model_names = os.listdir("./sample models/other")
    model_names = [m for m in model_names if m.split('.')[1] == "obj"]
    random_name = model_names[int(random.random()*len(model_names))]
    # path = f"./models/{random_name}"
    
    for n in model_names:
        path = f"./sample models/other/{n}"
        center,size = get_dimensions(path)
        if 0 in size:
            print(path)
    
    # path = "./sample models/other/1a5f561ce4cbca2625c70fb1df3f879b - Kopya.obj"
    
    # centers, size, faces = get_dimensions(path)
    
    # f = open(path)
    
    # lines = f.readlines()
    
    # faces_new = [fa-centers for fa in faces]
    
    # f_count = 0
    # for i in range(len(lines)):
    #     if lines[i].split(' ')[0] == 'v':
    #         lines[i] = "v " + arr2str(faces_new[f_count]/size) + '\n'
    #         f_count += 1
    
    # new_str = "".join(lines)
    
    # f_w = open("new.obj", 'w+')
    
    # f_w.write(new_str)




# scene = pywavefront.Wavefront(path, collect_faces = True)

# faces = np.array(scene.mesh_list[0].faces)

# centers = np.array([0,0,0])

# for i in range(3):
#     is_ = faces[:,i]
#     print(faces.shape)
#     i_center = (np.min(is_) + np.max(is_))/2
#     centers[i] = i_center

# print(centers)
  
# -*- coding: utf-8 -*-
"""
CENG501 - Spring 2021 
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
    
    for n in model_names:
        path = f"./sample models/other/{n}"
        center,size = get_dimensions(path)
        if 0 in size:
            print(path)
    

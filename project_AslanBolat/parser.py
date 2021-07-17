import numpy as np
import open3d as o3d

import json
import os
import subprocess
import multiprocessing
import pytorch3d
import torch
import trimesh
import numpy as np
from pytorch3d.ops import cubify, sample_points_from_meshes
from pytorch3d.vis.plotly_vis import plot_scene

import subprocess
import open3d as o3d

import binvox_rw
import copy
import os
import json

import time

dataset_path = "./dataset/"
processed_path = "./processed_np/"
split_path =  dataset_path + "train_test_split/"
class_name_list = ["03001627"] # "03001627": chair, "02691156": airplane, "03636649": lamp, "03790512": motorbike
instance_name_list = ["1ace72a88565df8e56bd8571ad86331a"]

def read_points_to_numpy(filename):
    points = []
    with open(filename) as pcf:
        for line in pcf:
            point = line.strip().split(" ")
            points.append(point)
    return np.array(points, dtype=np.float64)

def read_labels_to_numpy(filename):
    labels = []
    with open(filename) as lf:
        for line in lf:
            label = line.strip()
            labels.append(label)
    return np.array(labels, dtype=np.int32)

def numpy_to_pcs(labels_np, points_np):
    cloud_dict = {}
    max_label = labels_np.max()
    
    x = points_np[:, 0]
    y = points_np[:, 1]
    z = points_np[:, 2]


    axs0_max = x.argmax()
    axs1_max = y.argmax()
    axs2_max = z.argmax()
    axs0_min = x.argmin()
    axs1_min = y.argmin()
    axs2_min = z.argmin()
    
    for part_label in range(1, max_label+1):
        part_indicies = labels_np == part_label
        # part_indicies[axs0_max] = True
        # part_indicies[axs1_max] = True
        # part_indicies[axs2_max] = True
        # part_indicies[axs0_min] = True
        # part_indicies[axs1_min] = True
        # part_indicies[axs2_min] = True
        part_points = points_np[part_indicies]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part_points)
        cloud_dict[part_label]= pcd
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(points_np)
    cloud_dict[0] = pcd
    return cloud_dict

def pointcloud_to_binvox(pointcloud_dict, dirname, instancename):
    alpha = 0.070
    for key in pointcloud_dict.keys():
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pointcloud_dict[key], alpha)
        mesh.compute_vertex_normals()
        new_file_name = dirname+"/"+str(key)+"/"+instancename+".ply"
        o3d.io.write_triangle_mesh(new_file_name, mesh)
        subprocess.run(["./binvox","-e","-pb", "-d", "64", new_file_name])
        os.remove(new_file_name)

def process_data(split_filename, process_count = 8):
    split_file_path = split_path + split_filename
    set_name = "val/" if "val" in split_filename else "test/" if "test" in split_filename else "train/"
    filenames = None
    with open(split_file_path) as f: filenames = json.load(f)
    processes = []
    
    for i in range(process_count):
        p = multiprocessing.Process(target=multi_process, args=(filenames[int(i*len(filenames)/process_count):int((i+1)*len(filenames)/process_count)],set_name, ))
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()

def multi_process(filenames, set_name):
    for filename in filenames:
        tick = time.time()
        splits = filename.split("/")
        class_name, instance_name = splits[1], splits[2]
        # if not class_name in class_name_list or not instance_name in instance_name_list: continue
        if not class_name in class_name_list: continue
        new_dir_name = processed_path+set_name+class_name
        if not os.path.isdir(new_dir_name): os.mkdir(new_dir_name)
        pc_file_path = dataset_path + class_name + "/points/" + instance_name + ".pts"
        label_file_path = dataset_path + class_name + "/points_label/" + instance_name + ".seg"
        labels_np = read_labels_to_numpy(label_file_path)
        points_np = read_points_to_numpy(pc_file_path)
            
        points = torch.from_numpy(points_np)#.cuda()
        labels = torch.from_numpy(labels_np)#.cuda()

        max_p = points.max()
        min_p = points.min()

        vol_axes = [
            torch.linspace(min_p,max_p, 64, dtype=torch.float32, device="cpu")
            for _ in range(3) 
        ]  
        Z, Y, X = torch.meshgrid(vol_axes)
        vol_coords = torch.stack((X, Y, Z), dim=3)[None].repeat(1, 1, 1, 1, 1)#.cuda()

        part_list = []
        part_list.append(points)
        for part_label in range(1, 5):
            part_indicies = labels == part_label
            part_points = points[part_indicies]
            part_list.append(part_points)
        

        for part_id, part in enumerate(part_list, 0):
            if not os.path.isdir(new_dir_name+"/"+str(part_id)):os.mkdir(new_dir_name+"/"+str(part_id))
            voxels = torch.zeros((1,64,64,64), dtype=bool)              
            for point in part:
                point_r = point.view(1,1,1,1,3)
                n = torch.linalg.norm(vol_coords - point_r, dim=4)
                indicies= n < 1e-2
 
                voxels[indicies] = True
            np.save( new_dir_name+"/"+str(part_id)+"/"+instance_name+".npy",voxels.cpu().numpy())
        print(time.time()-tick)

if __name__ == "__main__":
    if not os.path.isdir(processed_path):os.mkdir(processed_path)
    if not os.path.isdir(processed_path+"train"):os.mkdir(processed_path+"train")
    if not os.path.isdir(processed_path+"val"):os.mkdir(processed_path+"val")
    if not os.path.isdir(processed_path+"test"):os.mkdir(processed_path+"test")
    train_split_filename = "shuffled_train_file_list.json"
    val_split_filename = "shuffled_val_file_list.json"
    test_split_filename = "shuffled_test_file_list.json"
    process_data(train_split_filename, process_count=8)
    process_data(val_split_filename, process_count=8)
    process_data(test_split_filename, process_count=8)
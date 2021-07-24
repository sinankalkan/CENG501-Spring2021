import pytorch3d
import torch
import trimesh
import numpy as np
from pytorch3d.ops import cubify, sample_points_from_meshes
from pytorch3d.vis.plotly_vis import plot_scene

import subprocess
import open3d as o3d

import copy
from parser import read_labels_to_numpy, read_points_to_numpy
import os
import json
import time
# device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "./dataset/"
processed_path = "./processed_np_final/"
split_path =  dataset_path + "train_test_split/"
class_name_list = ["03001627"] # "03001627": chair, "02691156": airplane, "03636649": lamp, "03790512": motorbike
instance_name_list = ["1ace72a88565df8e56bd8571ad86331a"]

def visualize_voxels_color(voxels_list, dim_size, thresh=0.0001, name="pointcloud", device="cpu",show=True):
    colors = [(85/255,220/255,100/255),(0,32/255,1),(149/255,85/255,0), (1,104/255,196/255)]
    points_list = []
    texture_list = []
    for index,voxel in enumerate(voxels_list):
        data = voxel.view(1, 64, 64, 64)
        volume = pytorch3d.structures.Volumes([data])
        grid = volume.get_coord_grid()
        volume= 2*data >= thresh
        points = grid[volume]
        red = colors[index][0]*torch.ones(points.shape[0],1, device=device)
        green = colors[index][1]*torch.ones(points.shape[0],1, device=device)
        blue = colors[index][2]*torch.ones(points.shape[0],1, device=device)
        texture = torch.cat([red,green,blue], dim=1)
        points_list.append(points)
        texture_list.append(texture)
    pointcloud = pytorch3d.structures.Pointclouds(points_list, features=texture_list) 
    figure=plot_scene({
        name: {
            "person": pointcloud
        }
    })
    if show:
        figure.show()
    else:
        figure.write_image("./images/{}.png".format(name))
def visualize_voxels(voxels, dim_size, thresh=0.0001, name="pointcloud", device="cpu",show=True):
    # visualize voxel data as pointclouds
    voxels = voxels.view(voxels.size(0), dim_size, dim_size, dim_size)
    for voxel in voxels:
        data = voxel.view(1, 64, 64, 64)
        volume = pytorch3d.structures.Volumes([data])
        grid = volume.get_coord_grid()
        volume= 2*data >= thresh
        points = grid[volume]
        textures = 0.5*torch.ones(1,points.shape[0],3, device=device)
        pointcloud = pytorch3d.structures.Pointclouds([points], features=textures) 
        figure=plot_scene({
            name: {
                "person": pointcloud
            }
        })
        if show:
            figure.show()
        else:
            figure.write_image("./images/{}.png".format(name))


def translate_voxels(x, translate):
    translateZ, translateY, translateX = translate
    padX = [translateX, 0] if translateX > 0 else [0, -translateX]
    padY = [translateY, 0] if translateY > 0 else [0, -translateY]
    padZ = [translateZ, 0] if translateZ > 0 else [0, -translateZ]
    pad_width = []
    pad_width.extend(padX)
    pad_width.extend(padY)
    pad_width.extend(padZ)
    # print(x)
    # print("#############")
    slice_x = slice(0, x.shape[0]) if translateX == 0 else slice(0, -translateX) if translateX > 0 else slice(-translateX, x.shape[0]-translateX)
    slice_y = slice(0, x.shape[1]) if translateY == 0 else slice(0, -translateY) if translateY > 0 else slice(-translateY, x.shape[1]-translateY)
    slice_z = slice(0, x.shape[2]) if translateZ == 0 else slice(0, -translateZ) if translateZ > 0 else slice(-translateZ, x.shape[2]-translateZ)
    indices = (slice_x, slice_y, slice_z)

    # x = (x, pad_width, mode='constant', constant_values=(False, False))[indices]
    x = torch.nn.functional.pad(x, pad_width, mode='constant', value=0)[indices]
    return x

def points_to_part_clouds(points_np, labels_np):
    points = torch.from_numpy(points_np).cuda()
    labels = torch.from_numpy(labels_np).cuda()

    part_list = []
    part_list.append(points)
    for part_label in range(1, 5):
        part_indicies = labels == part_label
        part_points = points[part_indicies]
        part_list.append(part_points)
   
    return part_list

def voxel_coordinates(min_p, max_p):
    vol_axes = [
        torch.linspace(min_p, max_p, 64, dtype=torch.float32, device="cuda") # linspace end dim is inclusive [0,63] #64
        for _ in range(3) 
    ]  
    Z, Y, X = torch.meshgrid(vol_axes)
    vol_coords = torch.stack((X, Y, Z), dim=3)[None].repeat(1, 1, 1, 1, 1).cuda()
    return vol_coords

def pointcloud_to_voxels(pointcloud, thresh, min_p=0, max_p=63):
    vol_coords = voxel_coordinates(min_p, max_p)
    voxels = torch.zeros((1,64,64,64), dtype=bool).cuda() 
    for point in pointcloud:
        point_r = point.view(1,1,1,1,3)
        n = torch.linalg.norm(vol_coords - point_r, dim=4)
        voxels[n < thresh] = True
    return voxels

def random_transform():
    transform = torch.eye(4).cuda()
    scale = torch.rand(1, device="cuda")/2
    scale += 0.75
    transform[:3,:3] *= scale
    transform[:3, 3] = torch.randint(-5, 5, (3,), device="cuda")
    return transform

def transform_pointcloud(pointcloud, transform):
    padded = torch.nn.functional.pad(pointcloud, (0,1), mode='constant', value=1)
    transpose = transform.transpose(0,1)
    transformed = torch.matmul(padded, transpose)
    # print(transformed[:5])
    # print(pointcloud[:5])
    return transformed[:,:3]

def voxels_to_pointcloud(voxels):
    vol_coords = voxel_coordinates(0,63)
    points = vol_coords[voxels.view(1,64,64,64).bool()]
    return points

def process_data_np(split_filename, filenames):
    
    set_name = "val/" if "val" in split_filename else "test/" if "test" in split_filename else "train/"
    show = 0
    for filename in filenames:
        tick = time.time()
        splits = filename.split("/")
        class_name, instance_name = splits[1], splits[2]
        # if not class_name in class_name_list or not instance_name in instance_name_list: continue
        if not class_name in class_name_list: continue
        new_dir_name = processed_path+set_name+class_name
        if not os.path.isdir(new_dir_name): os.mkdir(new_dir_name)
        for part_id in range(5):
            if not os.path.isdir(new_dir_name+"/"+str(part_id)):os.mkdir(new_dir_name+"/"+str(part_id)) 
        pc_file_path = dataset_path + class_name + "/points/" + instance_name + ".pts"
        label_file_path = dataset_path + class_name + "/points_label/" + instance_name + ".seg"
        
        labels_np = read_labels_to_numpy(label_file_path)
        points_np = read_points_to_numpy(pc_file_path)
        part_list = points_to_part_clouds(points_np, labels_np)
        min_p, max_p =  part_list[0].min(), part_list[0].max()
        p_dist = torch.linalg.norm(max_p - min_p)/63
        for part_id, part in enumerate(part_list, 0):
            if part.nelement() == 0: continue
            voxels = pointcloud_to_voxels(part, p_dist,min_p, max_p)
            # if show %10 == 0: 
            #     visualize_voxels(voxels, 64, device="cuda") 
            np.save( new_dir_name+"/"+str(part_id)+"/"+instance_name+".npy",voxels.cpu().numpy())
        show+=1
        # print(time.time()-tick)

def main_process(filenames):
    if not os.path.isdir(processed_path):os.mkdir(processed_path)
    if not os.path.isdir(processed_path+"train"):os.mkdir(processed_path+"train")
    if not os.path.isdir(processed_path+"val"):os.mkdir(processed_path+"val")
    if not os.path.isdir(processed_path+"test"):os.mkdir(processed_path+"test")
    train_split_filename = "shuffled_train_file_list.json"
    val_split_filename = "shuffled_val_file_list.json"
    test_split_filename = "shuffled_test_file_list.json"
    # process_data_np(train_split_filename, filenames)
    process_data_np(val_split_filename,filenames)
    # process_data_np(test_split_filename)

def main_transform(part_id):
    from dataset import PartNetDataset
    from torch.utils.data import  DataLoader
    partnetdataset = PartNetDataset("./dataset/train_test_split/shuffled_val_file_list.json",
                                "./processed_np_final/val/", "03001627", part_id)
    dataloader = DataLoader(partnetdataset, batch_size=1)
    # print(len(partnetdataset), len(partnetdataset.part_file_names))
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # print(i)
            filename = partnetdataset.part_file_names[i]
            # print(filename.split(".npy"))
            name, _ = filename.split(".npy")
            voxels = data.cuda()
            pointcloud = voxels_to_pointcloud(voxels)
            transformation = random_transform()
            transformed = transform_pointcloud(pointcloud, transformation)
            # print(torch.diag(transformation[:3,:3],0), transformation[:3, 3])
            # translation = torch.eye(4).cuda()
            # translation[:3, 3] = translate
            # translated = transform_pointcloud(pointcloud, translation)
            transformed_voxels = pointcloud_to_voxels(transformed, 0.866)
            np.save(name+"_transformed"+".npy",transformed_voxels.cpu().numpy())
            np.save(name+"_transformation"+".npy",transformation.cpu().numpy())
            if i == 1000: break
        
            # print(voxels.sum(), transformed_voxels.sum(), voxels.shape, transformed_voxels.shape)
            # visualize_voxels(voxels, 64, device="cuda")
            # visualize_voxels(transformed_voxels, 64, device="cuda")
    
           
def compute_retransformed_voxels(part_voxels, scale, translate):
    scale = scale.view(-1)
    part_voxels = part_voxels.view(-1, 64, 64, 64)
    translate = translate.view(-1,3)
    # print(scale.size(), translate.size())
    retransformed_voxel = torch.zeros((1,64,64,64))
    for index in range(4):
        pointcloud = voxels_to_pointcloud(part_voxels[index])
        transformation = torch.eye(4).cuda()
        transformation[:3,:3] *= scale[index].item()
        transformation[:3,3] = translate[index]
        # print(scale[index].item(), translate[index])
        transformed = transform_pointcloud(pointcloud, transformation)
        new_voxels = pointcloud_to_voxels(transformed, 0.8666)
        retransformed_voxel[new_voxels.view(1,64,64,64).bool()] = 1.0
    return retransformed_voxel

def compute_retransformed_voxels_list(part_voxels, scale, translate):
    scale = scale.view(-1)
    part_voxels = part_voxels.view(-1, 64, 64, 64)
    translate = translate.view(-1,3)
    # print(scale.size(), translate.size())
    retransformed_voxels_list = []
    for index in range(4):
        retransformed_voxel = torch.zeros((1,64,64,64))
        pointcloud = voxels_to_pointcloud(part_voxels[index])
        transformation = torch.eye(4).cuda()
        transformation[:3,:3] *= scale[index].item()
        transformation[:3,3] = translate[index]
        # print(scale[index].item(), translate[index])
        transformed = transform_pointcloud(pointcloud, transformation)
        new_voxels = pointcloud_to_voxels(transformed, 0.8666)
        retransformed_voxel[new_voxels.view(1,64,64,64).bool()] = 1.0
        retransformed_voxels_list.append(retransformed_voxel)
    return retransformed_voxels_list
if __name__ == "__main__":
    # filenames = None
    # split_file_path = split_path + "shuffled_val_file_list.json"
    # with open(split_file_path) as f: filenames = json.load(f)

    # main_process(filenames) 


    main_transform(1)
    # main_transform(2)
    # main_transform(3)
    # main_transform(4)





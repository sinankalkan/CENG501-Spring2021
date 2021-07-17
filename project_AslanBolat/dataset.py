from numpy import dtype
import torch
from torch.utils.data import Dataset

import json
import os

import numpy as np

file_path = "./dataset/train_test_split/shuffled_train_file_list.json"
processed_path = "./processed_np/train/"

class AssemblerDataset(Dataset): # batch_size, num_parts, dimx, dimy, dimz
    def __init__(self, file_path, processed_path, class_id, anchor_id, num_parts=4, num_deformed=700, num_orig=100):
        self.part_list = []
        self.regres_scale_list = []
        self.regres_translate_list = []
        count_orig = 0
        count_deform = 0
        file_names = None
        with open(file_path) as f: file_names = json.load(f)
        for file_name in file_names:
            splits = file_name.split("/")
            class_name, instance_name = splits[1], splits[2]  
            if not class_name == class_id: continue
            # lists for later concatanating part voxels of a model
            translate_list = []
            scale_list = []
            orig_list = []
            transformed_list = []
            
            transformotion_path = processed_path + class_name + "/" + str(anchor_id) + "/" + instance_name + "_transformation.npy"
            if not os.path.exists(transformotion_path): continue
            for part_id in range(1, num_parts+1):
                orig_path = processed_path + class_name + "/" + str(part_id) + "/" + instance_name + ".npy"
                transformed_path = processed_path + class_name + "/" + str(part_id) + "/" + instance_name + "_transformed.npy"
                transformotion_path = processed_path + class_name + "/" + str(part_id) + "/" + instance_name + "_transformation.npy"
                orig = None
                if os.path.exists(orig_path):
                    orig = torch.from_numpy(np.load(orig_path).astype('float32'))
                    if part_id == anchor_id:
                        orig_list.append(orig*-1)
                    else:
                        orig_list.append(orig)
                else:
                    if part_id == anchor_id:  # If the  work it work
                        break
                    else:
                        orig = torch.zeros((1,64,64,64))
                        orig_list.append(orig)

                if part_id == anchor_id:
                    transformed_list.append(orig*-1)
                    translate_list.append(torch.zeros((3,)))
                    scale_list.append(torch.ones((1,)))
                
                elif os.path.exists(transformed_path) and os.path.exists(transformotion_path):
                    transformed = torch.from_numpy(np.load(transformed_path).astype('float32'))
                    transformotion = torch.from_numpy(np.load(transformotion_path).astype('float32'))
                    scale = transformotion[0, 0]
                    translate = transformotion[:3,3]
                    transformed_list.append(transformed)
                    # the assembler learns the reverse of the transformations applied to the parts to assemble them
                    # transformation order: first scale then translate
                    translate = translate*(-1/scale) # reversing the translation 
                    translate_list.append(translate)
                    scale = 1/scale                 # reversing the scale
                    scale_list.append(scale.view(1,))

                else:
                    transformed_list.append(orig)
                    translate_list.append(torch.zeros((3,)))
                    scale_list.append(torch.ones((1,)))
                   
            if len(orig_list) == num_parts  and count_orig < num_orig:
                self.part_list.append(torch.cat(orig_list, dim=0))  # batch num_parts 64 64 64
                self.regres_scale_list.append(torch.ones((4,)))     # batch num_parts * 1
                self.regres_translate_list.append(torch.zeros(12,)) # batch num_parts * 3
                count_orig+=1
            if len(transformed_list) == num_parts and count_deform < num_deformed:
                self.part_list.append(torch.cat(transformed_list, dim=0))
                self.regres_scale_list.append(torch.cat(scale_list, dim=0))
                self.regres_translate_list.append(torch.cat(translate_list, dim=0))
                count_deform += 1       
       
    def __getitem__(self, index):
        return self.part_list[index], self.regres_scale_list[index], self.regres_translate_list[index]

    def __len__(self):
        return len(self.part_list)
        

class PartNetDataset(Dataset):
    def __init__(self, file_path, processed_path, class_id, part_id, num_deformed=300):
        self.part_list = []
        self.part_file_names = []
        file_names = None
        count_deformed = 0
        with open(file_path) as f: file_names = json.load(f)
        for file_name in file_names:
            splits = file_name.split("/")
            class_name, instance_name = splits[1], splits[2]  
            if not class_name == class_id: continue
            orig_path = processed_path + class_name + "/" + str(part_id) + "/" + instance_name + ".npy"
            transformed_path = processed_path + class_name + "/" + str(part_id) + "/" + instance_name + "_transformed.npy"
            transformotion_path = processed_path + class_name + "/" + str(part_id) + "/" + instance_name + "_transformation.npy"
            if os.path.exists(orig_path):
                self.part_file_names.append(orig_path)
                orig = np.load(orig_path).astype('float32')
                self.part_list.append(torch.from_numpy(orig)) # (batch, 1, 64, 64, 64)
            if count_deformed < num_deformed and os.path.exists(transformed_path) and os.path.exists(transformotion_path):
                transformed = np.load(transformed_path).astype('float32')
                transformotion = np.load(transformotion_path).astype('float32')
                scale = transformotion[0,0]
                if (scale > 1 and orig.sum() > transformed.sum()) or (scale < 1 and orig.sum() < transformed.sum()):
                    continue
                self.part_list.append(torch.from_numpy(transformed))
                count_deformed += 1

    def __getitem__(self, index):
        return self.part_list[index]

    def __len__(self):
        return len(self.part_list)


def get_part_datasets(file_path, processed_path, class_dict):
    part_dataset_dict = {}
    for class_name in class_dict.keys():
        part_dataset_dict[class_name]= {}
        for index in range(class_dict[class_name]):
            part_dataset_dict[class_name][index] = PartNetDataset(file_path, processed_path, class_name, index)
    return part_dataset_dict

if __name__ == "__main__":
    file_path = "./dataset/train_test_split/shuffled_train_file_list.json"
    processed_path = "./processed_np_final/train/"
    from utils import visualize_voxels
    # class_dict = {"03001627" : 4}
    # part_datasets = get_part_datasets(file_path, processed_path, class_dict)
    # print(len(part_datasets["03001627"][0]))
    ad = AssemblerDataset(file_path, processed_path, "03001627", 1, num_deformed=200, num_orig=0)
    j = 0
    print("dataset size:", len(ad))
    # for data, scale, translate in ad:
    #     # visualize_voxels(data, 64)
    #     if j > 100 and j%10 == 0:
    #         new_voxel = torch.zeros((1,64,64,64))
    #         for i in range(4):
    #             new_voxel[data[i].view(1,64,64,64).bool()] = 1.0 
    #         visualize_voxels(new_voxel, 64)
    #         print(scale)
    #         print(translate)

    #         print(data.shape, scale.shape, translate.shape)
            
    #     j+=1
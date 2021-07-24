import torch
import torchviz
import torch.nn.functional as F
from networks import VAE, Assembler
from utils import visualize_voxels
from torch.utils.data import DataLoader
from dataset import AssemblerDataset
import utils
from utils import compute_retransformed_voxels, compute_retransformed_voxels_list

device = "cuda" if torch.cuda.is_available() else "cpu"

def assemble_parts(parts_list, anchor_id, device):
    channel_sizes = [64, 128, 256, 512, 100]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50
    num_parts = 4 
    assembler = Assembler(num_parts, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)

    assembler.load_state_dict(torch.load("./models/trained_assembler/assembler_final_logcosh_anchor{}_199.pt".format(anchor_id)))
    assembler.to(device)
    assembler.eval()
    retransformed_voxels = None
    with torch.no_grad():
        voxels = torch.cat(parts_list, dim=1)  > 1e-1
        voxels = voxels.float()
        predicted_translate, predicted_scale = assembler(voxels.to(device))
        transformed_voxels_list = compute_retransformed_voxels_list(voxels, predicted_scale, predicted_translate)
    return transformed_voxels_list

def voxelIoU(predicted_voxel, target_voxel):
    generated_voxel = predicted_voxel > 5e-1
    target_voxel = target_voxel.bool()
    intersection = torch.mul(generated_voxel, target_voxel).sum().item()
    union = (generated_voxel + target_voxel) >= 1
    union = union.sum().item()
    return intersection/union

def evaluate_assembler(model, dataloader):
    model.eval()
    mse = 0
    iou = 0
    total = 0
    with torch.no_grad():
        for i, (voxels, target_scale, target_translate) in enumerate(dataloader):
            predicted_translate, predicted_scale = model(voxels.to(device))
            translate_loss = F.mse_loss(predicted_translate, target_translate.to(device)) 
            scale_loss = F.mse_loss(predicted_scale, target_scale.to(device))
            mse += (translate_loss.item()+ scale_loss.item())
            total +=1
            retransformed_voxels = compute_retransformed_voxels(voxels, predicted_scale, predicted_translate)
            correct_assembled_voxels = compute_retransformed_voxels(voxels, target_scale, target_translate)
            iou += voxelIoU(retransformed_voxels, correct_assembled_voxels)
            visualize_voxels(retransformed_voxels, 64, name="predicted_assembly{}".format(i), show=False)
            visualize_voxels(correct_assembled_voxels, 64, name="correct_assembly{}".format(i), show=False)

            # visualize retransformed model
            # if target_translate.sum() != 0 or target_scale.sum() != 3:
            #     retransformed_voxels = compute_retransformed_voxels(voxels, predicted_scale, predicted_translate)
            #     visualize_voxels(retransformed_voxels, 64)

    average_mse = mse/float(total)
    average_iou = iou/float(total)
    print("Assembler average mse over dataset:",average_mse)
    print("Average iou of transformed and target voxels:",average_iou)

    return average_mse, average_iou


def assembler_main():
    channel_sizes = [64, 128, 256, 512, 100]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50
    num_parts = 4 
    for anchor_id in range(1,5):
        assembler_dataset = AssemblerDataset("./dataset/train_test_split/shuffled_val_file_list.json",
                                            "./processed_np_final/val/", "03001627", anchor_id, 
                                            num_deformed=100, num_orig=0)
        assembler_dataloader = DataLoader(assembler_dataset)
        assembler = Assembler(num_parts, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)

        assembler.load_state_dict(torch.load("./models/trained_assembler/assembler_final_logcosh_anchor{}_199.pt".format(anchor_id)))
        print("Anchor", anchor_id)
        evaluate_assembler(assembler, assembler_dataloader)
        
if __name__ == "__main__":
    assembler_main()


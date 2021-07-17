import torch
import torchviz
import torch.nn.functional as F
from networks import VAE, Assembler
from utils import visualize_voxels
from torch.utils.data import DataLoader
from dataset import AssemblerDataset
import utils
from utils import compute_retransformed_voxels

device = "cuda" if torch.cuda.is_available() else "cpu"
def evaluate_assembler(model, dataloader):
    model.eval()
    mse = 0
    total = 0
    with torch.no_grad():
        for i, (voxels, target_scale, target_translate) in enumerate(dataloader):
            predicted_translate, predicted_scale = model(voxels.to(device))
            translate_loss = F.mse_loss(predicted_translate, target_translate.to(device)) 
            scale_loss = F.mse_loss(predicted_scale, target_scale.to(device))
            mse += (translate_loss.item()+ scale_loss.item())
            total +=1
            # visualize retransformed model
            if target_translate.sum() != 0 or target_scale.sum() != 3:
                retransformed_voxels = compute_retransformed_voxels(voxels, predicted_scale, predicted_translate)
                visualize_voxels(retransformed_voxels, 64)

    average_mse = mse/float(total)
    print("Assembler average mse over dataset:",average_mse)
    return average_mse



if __name__ == "__main__":
    channel_sizes = [64, 128, 256, 512, 100]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50
    num_parts = 4 
    assembler_dataset = AssemblerDataset("./dataset/train_test_split/shuffled_val_file_list.json",
                                         "./processed_np_final/val/", "03001627", 1, 
                                         num_deformed=20, num_orig=0)
    assembler_dataloader = DataLoader(assembler_dataset)
    print(len(assembler_dataset))
    assembler = Assembler(num_parts, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)

    assembler.load_state_dict(torch.load("./models/trained_assembler/assembler_final_logcosh199.pt"))
    evaluate_assembler(assembler, assembler_dataloader)


from dataset import PartNetDataset
from torch.utils.data import  DataLoader
import matplotlib as plt
from networks import Encoder, Decoder, Discriminator, VAE
from utils import visualize_voxels
import torch

import binvox_rw

def validate_vae(model, dataloader,device, part_id, rec_loss, random_index = 0):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            voxels = data.to(device)
            if i == random_index:
                mu, log_sigma, fake_voxels = model(voxels)
                mu = mu[0].view(-1,50)
                mu_voxel = model.decoder(mu) 
                if rec_loss == "mse":
                    visualize_voxels(voxels, voxels.size(-1), thresh=1e-2 ,name="mu_{}_{}_real".format(part_id, i), device=device, show=False)
                visualize_voxels(mu_voxel, mu_voxel.size(-1),thresh=5e-2, name="mu_{}_{}_{}".format(part_id,i, rec_loss), device=device, show=False)
                break

def main():
    device="cpu"
    channel_sizes = [64, 128, 256, 512, 100]
    channel_sizes_decoder = [512, 256, 128, 64, 1]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50
    for _ in range(5):
        for part_id in range(1,5):
            partnetdataset = PartNetDataset("./dataset/train_test_split/shuffled_val_file_list.json",
                                    "./processed_np_final/val/", "03001627", part_id,num_deformed=0)
            dataset_size =  len(partnetdataset)
            random_index = torch.randint(0,dataset_size-1, (1,)).item()
            for rec_loss in ["mse", "bce"]:
                model = VAE(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
                model.load_state_dict(torch.load("./models/trained_vaegan/part{}/vae_final_part{}_{}_dc_20.pt".format(part_id,part_id, rec_loss)))
                dataloader = DataLoader(partnetdataset, batch_size=1)
                validate_vae(model, dataloader, device, part_id, rec_loss, random_index)

if __name__ == "__main__":
    main()

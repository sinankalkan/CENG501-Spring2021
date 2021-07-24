from dataset import PartNetDataset
from torch.utils.data import  DataLoader
import matplotlib as plt
from networks import Encoder, Decoder, Discriminator, VAE
from utils import visualize_voxels
import torch
from test_assembler import voxelIoU

def check_ref_symmetry(voxels, thresh, device, part_id, rec_loss, i):
    voxels = voxels > thresh
    voxels = voxels.float()
    left_voxels = torch.zeros(voxels.shape)
    right_voxels = torch.zeros(voxels.shape)
    x = voxels.nonzero(as_tuple=False)
    li = x[:,2] <= torch.median(x[:,2])
    ri = x[:,2] > torch.median(x[:,2])
    li = x[li]
    li = (li[:,0], li[:,1], li[:,2], li[:,3], li[:,4])
    ri = x[ri]
    ri = (ri[:,0], ri[:,1], ri[:,2], ri[:,3], ri[:,4])
    left_voxels[li] = 1
    right_voxels[ri] = 1
    left_voxels = torch.flip(left_voxels,[2])
    # visualize_voxels(voxels, voxels.size(-1), thresh=2e-2 ,name="mu_{}_{}_real".format(part_id, i), device=device, show=True)
    # visualize_voxels(left_voxels, left_voxels.size(-1),thresh=5e-2, name="left_{}_{}_{}".format(part_id,i, rec_loss), device=device, show=True)
    # visualize_voxels(right_voxels, right_voxels.size(-1),thresh=5e-2, name="right_{}_{}_{}".format(part_id,i, rec_loss), device=device, show=True)
    return voxelIoU(left_voxels, right_voxels)

def random_generation(device, part_id, rec_loss):
    channel_sizes = [64, 128, 256, 512, 100]
    channel_sizes_decoder = [512, 256, 128, 64, 1]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50   
    num_gen = 1
    model = VAE(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
    model.load_state_dict(torch.load("./models/trained_vaegan/part{}/vae_final_part{}_{}_dc_20.pt".format(part_id,part_id, rec_loss)))
    model.eval()
    rand_voxel = None
    with torch.no_grad():
        rand = torch.normal(0,1,(50,), device=device)
        rand_voxel = model.decoder(rand)
    return rand_voxel

def validate_ref_symm(model, dataloader, device, part_id, rec_loss):
    model.eval()
    total = 0
    iou = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            
            if i == 1000:
                break
            total +=1
            voxels = data.to(device)
            mu, log_sigma, fake_voxels = model(voxels)
            mu = mu[0].view(-1,50)
            mu_voxel = model.decoder(mu)
            iou += check_ref_symmetry(mu_voxel, 2e-2, device, part_id, rec_loss, i) 
    print(total)
    return iou/total


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

def visual_val_main():
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
            for rec_loss in ["bce"]:
                model = VAE(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
                model.load_state_dict(torch.load("./models/trained_vaegan/part{}/vae_final_part{}_{}_dc_20.pt".format(part_id,part_id, rec_loss)))
                dataloader = DataLoader(partnetdataset, batch_size=1)
                validate_vae(model, dataloader, device, part_id, rec_loss, random_index)

def random_generation_main():
    from test_assembler import assemble_parts
    from utils import visualize_voxels_color
    device="cuda"
    rec_loss = "bce"
    anchor_id = 1
    for i in range(5):
        parts_list = []
        for part_id in range(1,5):
            rand_part = random_generation(device, part_id, rec_loss)
            parts_list.append(rand_part)
        assembled_parts_list = assemble_parts(parts_list, anchor_id, device)
        visualize_voxels_color(assembled_parts_list, 64, thresh=2e-2, name ="random_generated{}".format(i), show=False)

def reflective_main():
    device="cuda"
    channel_sizes = [64, 128, 256, 512, 100]
    channel_sizes_decoder = [512, 256, 128, 64, 1]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50
    for part_id in range(1,5):
        partnetdataset = PartNetDataset("./dataset/train_test_split/shuffled_train_file_list.json",
                                "./processed_np_final/train/", "03001627", part_id, num_deformed=1000)
        for rec_loss in ["bce"]:
            model = VAE(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
            model.load_state_dict(torch.load("./models/trained_vaegan/part{}/vae_final_part{}_{}_dc_20.pt".format(part_id,part_id, rec_loss)))
            dataloader = DataLoader(partnetdataset, batch_size=1)
            average_iou = validate_ref_symm(model, dataloader, device, part_id, rec_loss)
            print("Part", part_id, "Symmetry Average IoU", average_iou)

if __name__ == "__main__":
    reflective_main()

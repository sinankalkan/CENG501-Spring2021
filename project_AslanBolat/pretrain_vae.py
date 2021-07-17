import torch
import torchviz
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import PartNetDataset, get_part_datasets

import numpy as np
import matplotlib.pyplot as plt

from networks import Encoder, Decoder, Discriminator, VAE

torch.backends.cudnn.benchmark = True
device = None

def save_loss_plot(train_loss_history, part_id, loss_names):
    plt.plot(train_loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig("plots/pre_vae_loss_part{}_{}.png".format(part_id,loss_names))
    plt.clf()

def multivariate_kl_gauss_loss(mu, log_sigma):
    return torch.mean(0.5 * torch.sum((torch.exp(log_sigma) + torch.square(mu) - 1 -log_sigma), dim=1))

def pretrain_vae(model, optimizer, epochs, dataloader, part_id, kl_wait_interval=10, 
                  print_interval=1, save_interval=2):
    train_loss_history = []
    model.train()
    alpha1 = 10
    alpha2 = 2e-3
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            x = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, log_sigma, x_tilda = model(x)
            loss = alpha1* F.binary_cross_entropy(x_tilda, x, reduction="mean")
            # print("reconstruction loss:", loss.item())
            if epoch >= kl_wait_interval:
                kl_loss = alpha2 * multivariate_kl_gauss_loss(mu, log_sigma)
                # print("regularization loss:", kl_loss.item())
                loss = loss + kl_loss
            train_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()

        if True and (epoch % print_interval == 0 or epoch+1 == epochs): print(f'Epoch {epoch if epoch < epochs-1 else epochs} / {epochs}: avg. loss of last 5 iterations {np.sum(train_loss_history[:-6:-1])/5}')
        torch.save(model.state_dict(), "./models/pretrained_vae/part{}/pre_vae_final_part{}_bce20_kl10.pt".format(part_id, part_id))

    return train_loss_history


if __name__=="__main__":


    if torch.cuda.is_available():
        print("Cuda (GPU support) is available and enabled!")
        device = torch.device("cuda")
    else:
        print("Cuda (GPU support) is not available :(")
        device = torch.device("cpu")

    for i in range(1,4):
        channel_sizes = [64, 128, 256, 512, 100]
        channel_sizes_decoder = [512, 256, 128, 64, 1]
        kernel_sizes = [4, 4, 4, 4, 4]
        stride_sizes = [2, 2, 2, 2, 1]
        padding_sizes = [1, 1, 1, 1, 0]
        latent_size = 50

        partnetdataset = PartNetDataset("./dataset/train_test_split/shuffled_train_file_list.json",
                                        "./processed_np_final/train/", "03001627", i)
        dataloader = DataLoader(partnetdataset, batch_size=32, shuffle=True)

        vae = VAE(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
        # vae.load_state_dict(torch.load("./models/pretrained_vae/part1/pre_vae_final_part1_mse_kl.pt"))
        optimizer = optim.Adam(vae.parameters(), 0.001, (0.5, 0.999))
        train_loss_history = pretrain_vae(vae, optimizer, 30, dataloader, i, kl_wait_interval=20)
        save_loss_plot(train_loss_history, i, "bce20kl10")

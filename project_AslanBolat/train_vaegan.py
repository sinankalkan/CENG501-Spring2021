import torch
import torchviz
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import PartNetDataset, get_part_datasets

import matplotlib.pyplot as plt

from networks import Encoder, Decoder, Discriminator, VAE

from utils import visualize_voxels

torch.backends.cudnn.benchmark = True

device = None
if torch.cuda.is_available():
    print("Cuda (GPU support) is available and enabled!")
    device = torch.device("cuda")
else:
    print("Cuda (GPU support) is not available :(")
    device = torch.device("cpu")

def multivariate_kl_gauss_loss(mu, log_sigma):
    return torch.mean(0.5 * torch.sum((torch.exp(log_sigma) + torch.square(mu) - 1 -log_sigma), dim=1))
    
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP
        Taken from:
            https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1, 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), requires_grad=False, device=device, dtype=torch.float32)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgan_gp(model_dict, epochs, dataloader, part_id,
          n_critic=5, print_interval=1, save_interval=2, patience=2):
    
    discriminator, optimizer_disc = model_dict["discriminator"]
    vae, optimizer_vae = model_dict["vae"]
    data_size = model_dict["data_size"]
    recon_loss_name = model_dict["recon_loss"]
    alpha_one = 2e-3 if recon_loss_name == "bce" else 2e-4
    recon_loss_func = F.binary_cross_entropy if recon_loss_name =="bce" else F.mse_loss

    train_loss_history = []
    vae_loss_history = []

    vae.train()
    discriminator.train()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            vae.zero_grad()
            discriminator.zero_grad()

            real_volume = data.to(device)

            # ################### #
            # Train Discriminator #
            # ################### #

            last_disc_error = 0.0
            for j in range(n_critic):

                discriminator.zero_grad()
                vae.zero_grad()

                z = torch.Tensor(torch.normal(0, 1, (real_volume.shape[0], vae.latent_size))).to(device)  # Noise for generator
                fake_volume = vae.decoder(z).detach()  # detach not to train generator

                # Validations for real and fake images
                real_validity = discriminator(real_volume)
                fake_validity = discriminator(fake_volume)
                print("disc:", real_validity.sum().item(), fake_validity.sum().item())
                # Loss for Wasserstein GANs
                lambda_gp = 10  # Same as proposed at the paper "Improved Training of Wasserstein GANs"
                gradient_penalty = compute_gradient_penalty(discriminator, real_volume, fake_volume)
                wgan_loss = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gradient_penalty
                wgan_loss *= 1e-1
                last_disc_error = wgan_loss.item()  # Save only the last error of n_critic for the batch

                wgan_loss.backward()
                optimizer_disc.step()  # note that generator is detached, step will not affect it
        
            train_loss_history.append(last_disc_error)

            # ################### #
            # Train Generator     #
            # ################### #
        
            # Note that there are two generated fake volumes.
            # fake_volume_random + fake_volume_encoder
            # Adverserial loss  + reconstruction loss (mse)
            # D(G(zt)) + ||G(E(yi)) − xi||_2. 
            # Equation is taken from "3D-GAN" paper

            mu, log_sigma, fake_volume = vae(real_volume)

            # The following code is commented since it slows down the training by %6.32

            # z = torch.Tensor(torch.normal(0, 1, (real_volume.shape[0], vae.latent_size))).to(device)  # Noise for generator
            # fake_volume_random = vae.decoder(z)
            # fake_validity = discriminator(fake_volume_random)
            
            fake_validity = discriminator(fake_volume)
            print("gen:", fake_validity.sum().item())
            adv_loss = -torch.mean(fake_validity)
            kl_loss = multivariate_kl_gauss_loss(mu, log_sigma)
            recon_loss = recon_loss_func(fake_volume, real_volume)
            
            
            alpha_two = 1e-2

            vae_loss = alpha_one*kl_loss + alpha_two*adv_loss + recon_loss  

            vae.zero_grad()
            vae_loss_history.append(vae_loss.item())
            vae_loss.backward()
            optimizer_vae.step()
        
    
            if i % print_interval == 0: 
                print("Part {} Discriminator loss:".format(part_id), train_loss_history[-1], "epoch:", epoch, "%", (i*32)/data_size*100)
                print("Part {} VAE loss:".format(part_id), vae_loss_history[-1], "epoch:", epoch, "%", (i*32)/data_size*100)
        
        
        torch.save(vae.state_dict(), "./models/trained_vaegan/part{}/vae_final_part{}_{}.pt".format(part_id, part_id, recon_loss_name))
        torch.save(discriminator.state_dict(), "./models/trained_vaegan/part{}/discriminator_final_part{}_{}.pt".format(part_id, part_id, recon_loss_name))
                 
    plt.plot(vae_loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig("plots/vae_loss_part{}_{}.png".format(part_id,recon_loss_name))
    plt.clf()

    plt.plot(train_loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig("plots/disc_loss_part{}_{}.png".format(part_id,recon_loss_name))
    plt.clf()

    return train_loss_history

def train_dcgan(model_dict, epochs, dataloader, part_id,
          n_critic=5, print_interval=1, save_interval=2, patience=2):    
    discriminator, optimizer_disc = model_dict["discriminator"]
    vae, optimizer_vae = model_dict["vae"]
    data_size = model_dict["data_size"]
    recon_loss_name = model_dict["recon_loss"]
    alpha_one = 2e-3 if recon_loss_name == "bce" else 2e-4
    recon_loss_func = F.binary_cross_entropy if recon_loss_name =="bce" else F.mse_loss

    train_loss_history = []
    vae_loss_history = []

    vae.train()
    discriminator.train()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            vae.zero_grad()
            discriminator.zero_grad()

            real_volume = data.to(device)

            # ################### #
            # Train Discriminator #
            # ################### #

            discriminator.zero_grad()
            vae.zero_grad()

            # Validations for real images
            real_validity = discriminator(real_volume)
            
            b_size = real_volume.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            real_validity = real_validity.view(-1)
            # Loss for DCGANs
            error_real = F.binary_cross_entropy(real_validity, label)
            
            error_real.backward()
            optimizer_disc.step()  
        
            label.fill_(0)
            z = torch.Tensor(torch.normal(0, 1, (real_volume.shape[0], vae.latent_size))).to(device)  # Noise for generator
            fake_volume = vae.decoder(z).detach()  # detach not to train generator
            fake_validity = discriminator(fake_volume).view(-1)
            error_fake = F.binary_cross_entropy(fake_validity, label)
            error_fake.backward()
            optimizer_disc.step() # note that generator is detached, step will not affect it

            disc_error = (error_real + error_fake).item()  
            train_loss_history.append(disc_error)
            # ################### #
            # Train Generator     #
            # ################### #
            vae.zero_grad()

            label.fill_(1)
            mu, log_sigma, fake_volume = vae(real_volume)
            
            fake_validity = discriminator(fake_volume).view(-1)
            gen_error = F.binary_cross_entropy(fake_validity, label)
            kl_loss = multivariate_kl_gauss_loss(mu, log_sigma)
            recon_loss = recon_loss_func(fake_volume, real_volume)
            
            
            alpha_two = 5e-3

            vae_loss = alpha_one*kl_loss + alpha_two*gen_error + 10*recon_loss  

            
            vae_loss_history.append(vae_loss.item())
            vae_loss.backward()
            optimizer_vae.step()
        
    
            if i % print_interval == 0: 
                print("Part {} Discriminator loss:".format(part_id), train_loss_history[-1], "epoch:", epoch, "%", (i*32)/data_size*100)
                print("Part {} VAE loss:".format(part_id), vae_loss_history[-1], "epoch:", epoch, "%", (i*32)/data_size*100)
        
        
        torch.save(vae.state_dict(), "./models/trained_vaegan/part{}/vae_final_part{}_{}_dc_20.pt".format(part_id, part_id, recon_loss_name))
        torch.save(discriminator.state_dict(), "./models/trained_vaegan/part{}/discriminator_final_part{}_{}_dc_20.pt".format(part_id, part_id, recon_loss_name))
                 
    plt.plot(vae_loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig("plots/vae_loss_part{}_{}_20.png".format(part_id,recon_loss_name))
    plt.clf()

    plt.plot(train_loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig("plots/disc_loss_part{}_{}_20.png".format(part_id,recon_loss_name))
    plt.clf()
    return train_loss_history

if __name__=="__main__":
    channel_sizes = [64, 128, 256, 512, 100]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    latent_size = 50

    for recon_loss in ["mse", "bce"]:
        for part_id in range(1,5):
            dataset = PartNetDataset("./dataset/train_test_split/shuffled_train_file_list.json",
                                            "./processed_np_final/train/", "03001627", part_id)
            data_size = len(dataset)
            dataloader = DataLoader(dataset, batch_size=32)

            discriminator = Discriminator(channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
            vae = VAE(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
            vae.load_state_dict(torch.load("./models/pretrained_vae/part{}/pre_vae_final_part{}_{}20_kl10.pt".format(part_id, part_id, recon_loss)))

            optimizer_vae = optim.Adam(vae.parameters(), 0.001, [0.5, 0.999])
            optimizer_disc = optim.Adam(discriminator.parameters(), 0.001, [0.5, 0.999])
            model_dict = {"discriminator": (discriminator, optimizer_disc), 
                        "vae": (vae, optimizer_vae), 
                        "data_size": data_size, 
                        "recon_loss": recon_loss}
            train_dcgan(model_dict, 20, dataloader, part_id)








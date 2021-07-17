import torch
from torch import nn
from torchinfo import summary
import copy
device = "cuda" if torch.cuda.is_available() else "cpu"

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu= nn.ReLU()

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.relu(X)
        return X

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DecoderBlock, self).__init__()
        self.conv_trans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, X):
        X = self.conv_trans(X)
        X = self.bn(X)
        X = self.relu(X)
        return X

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu= nn.ReLU()

    def forward(self, X):
        X = self.conv(X)
        X = torch.layer_norm(X, X.size()[1:])
        X = self.relu(X)
        return X

class Encoder(nn.Module):
    def __init__(self, latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes):
        super(Encoder, self).__init__()
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channel_sizes):
            layers.append(EncoderBlock(in_channels, 
                                       out_channels, 
                                       kernel_sizes[index],
                                       stride_sizes[index],
                                       padding_sizes[index]))
            in_channels = out_channels 
        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Linear(100, latent_size)
        self.fc2 = nn.Linear(100, latent_size)

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        mu = self.fc1(X)
        log_sigma = self.fc2(X)
        return mu, log_sigma

    def reparametrize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.normal(0,1,std.shape).to(device)
        z = eps*std + mu
        return z

class Decoder(nn.Module):
    def __init__(self, latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        layers = []
        in_channels = latent_size
        for index, out_channels in enumerate(channel_sizes):
            if out_channels == 1:
                layers.append(nn.Sequential(nn.ConvTranspose3d(in_channels, 
                                                               out_channels, 
                                                               kernel_sizes[index],
                                                               stride_sizes[index],
                                                               padding_sizes[index]),
                                                               nn.Sigmoid()))
            else:
                layers.append(DecoderBlock(in_channels, 
                                           out_channels, 
                                           kernel_sizes[index],
                                           stride_sizes[index],
                                           padding_sizes[index]))
            in_channels = out_channels 
    
        self.deconv = nn.Sequential(*layers)
    def forward(self, X):
        X = X.view(-1, self.latent_size, 1, 1, 1)
        X = self.deconv(X)
        return X

class VAE(nn.Module):
    def __init__(self, latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes, training=True):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.training = training
        self.encoder = Encoder(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes)
        channel_sizes = channel_sizes[:-1] 
        channel_sizes = channel_sizes[::-1]
        channel_sizes.append(1)
        stride_sizes = stride_sizes[::-1]
        padding_sizes = padding_sizes[::-1]
        self.decoder = Decoder(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes)

    def reparametrize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.normal(0,1,std.shape, device=device)
        z = eps*std + mu
        return z
        
    def forward(self, X):
        mu, log_sigma = self.encoder(X)
        if self.training:
            z = self.reparametrize(mu, log_sigma)
            x_tilda = self.decoder(z)
        else:
            x_tilda = self.decoder(mu)
        return mu, log_sigma, x_tilda

class Discriminator(nn.Module):
    def __init__(self, channel_sizes, kernel_sizes, stride_sizes, padding_sizes):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1

        for index, out_channels in enumerate(channel_sizes):
            self.layers.append(DiscriminatorBlock(in_channels,
                                             out_channels, 
                                             kernel_sizes[index],
                                             stride_sizes[index],
                                             padding_sizes[index]))
            in_channels = out_channels 

        self.mlp = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X = X.view(X.size(0), -1)
        X = self.mlp(X)
        return X


class Assembler(nn.Module): #write and train
    def __init__(self, num_parts, channel_sizes, kernel_sizes, stride_sizes, padding_sizes):
        super(Assembler, self).__init__()
        layers = []
        in_channels = num_parts
        for index, out_channels in enumerate(channel_sizes):
            layers.append(EncoderBlock(in_channels, 
                                       out_channels, 
                                       kernel_sizes[index],
                                       stride_sizes[index],
                                       padding_sizes[index]))
            in_channels = out_channels 
        self.conv = nn.Sequential(*layers)
        self.fc1 = nn.Linear(100, 3*num_parts) # to regress part translations
        self.fc2 = nn.Linear(100, 1*num_parts) # to regress part scaling

    def forward(self, X):
        X = self.conv(X)
        X = X.view(X.size(0), -1)
        translations = self.fc1(X)
        scales = self.fc2(X)
        return translations, scales


if __name__ == "__main__":
    from dataset import AssemblerDataset
    from torch.utils.data import DataLoader
    file_path = "./dataset/train_test_split/shuffled_train_file_list.json"
    processed_path = "./processed_np/train/"
    
    assembler_dataset = AssemblerDataset(file_path, processed_path, "03001627", 1)
    dataloader = DataLoader(AssemblerDataset, batch_size=32)
    # for data in dataloader:

    channel_sizes = [64, 128, 256, 512, 100]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    num_parts = 4
    assembler = Assembler(num_parts, channel_sizes, kernel_sizes, stride_sizes, padding_sizes)
    X = torch.rand((10,num_parts,64,64,64)) # batch_size, num_parts, dims
    translations, scales = assembler(X)
    print(translations.size(), scales.size())
    # encoder = Encoder(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes)
    # X = torch.rand((10,1,64,64,64)) # batch_size, channel_size, dims
    # mu, log_sigma = encoder(X)
    # print(mu.size(), log_sigma.size())

    # channel_sizes = [512, 256, 128, 64, 1]
    # kernel_sizes = [4, 4, 4, 4, 4]
    # stride_sizes = [1, 2, 2, 2, 2]
    # padding_sizes = [0, 1, 1, 1, 1]
    # latent_size = 50
    # decoder = Decoder(latent_size, channel_sizes, kernel_sizes, stride_sizes, padding_sizes)

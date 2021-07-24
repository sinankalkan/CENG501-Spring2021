import torch
import torchviz
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import AssemblerDataset
import matplotlib.pyplot as plt

from networks import Assembler

from utils import visualize_voxels

torch.backends.cudnn.benchmark = True

device = None
if torch.cuda.is_available():
    print("Cuda (GPU support) is available and enabled!")
    device = torch.device("cuda")
else:
    print("Cuda (GPU support) is not available :(")
    device = torch.device("cpu")

def log_cosh_loss(predicted, target):
    return torch.sum(torch.log(torch.cosh(predicted-target)))
def train(model, epochs, dataloader, optimizer, anchor_id,
            print_interval=1, save_interval=2, patience=2):

    model.train()
    loss_history = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):

            volume_data, regres_scale_list, regres_translate_list = data
            volume_data = volume_data.to(device)
            regres_scale_list = regres_scale_list.to(device)
            regres_translate_list = regres_translate_list.to(device)
            translation, scaling = model(volume_data)

            # translation_loss = F.mse_loss(translation, regres_translate_list)            
            # scaling_loss = F.mse_loss(scaling, regres_scale_list)
            translation_loss =  log_cosh_loss(translation, regres_translate_list)
            scaling_loss = log_cosh_loss(scaling, regres_scale_list)

            loss = translation_loss + scaling_loss
            loss_history.append(loss.item())
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss_history[-1], epoch)
        if epoch % 9 == 0 or epoch +1 == epochs:
            torch.save(model.state_dict(), "./models/trained_assembler/assembler_final_logcosh_anchor{}_{}.pt".format(anchor_id,epoch))

    return loss_history

if __name__=="__main__":
    channel_sizes = [64, 128, 256, 512, 100]
    kernel_sizes = [4, 4, 4, 4, 4]
    stride_sizes = [2, 2, 2, 2, 1]
    padding_sizes = [1, 1, 1, 1, 0]
    for anchor_id in range(1,5):
        file_path = "./dataset/train_test_split/shuffled_train_file_list.json"
        processed_path = "./processed_np_final/train/"
        dataset = AssemblerDataset(file_path, processed_path, "03001627", anchor_id, 
                                    num_deformed=1000, num_orig=50)
        dataloader = DataLoader(dataset, batch_size=32)
        
        model = Assembler(4, channel_sizes, kernel_sizes, stride_sizes, padding_sizes).to(device)
        optimizer = optim.Adam(model.parameters(), 0.001, [0.5, 0.999])

        train_loss_history = train(model, 200, dataloader, optimizer, anchor_id)

        plt.plot(train_loss_history)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.savefig("plots/assembler_loss_logcosh_anchor{}.png".format(anchor_id))
        plt.clf()
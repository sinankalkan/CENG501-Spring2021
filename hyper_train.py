import optuna
import torch
from loss.mvrloss import MVR_Proxy, MVR_Triplet, MVR_MS, MVR_MS_reg
import torchvision.transforms as trsfrm
from dataloader.trsfrms import must_transform
from model.bn_inception import bn_inception
from dataloader.cub_dataset import CUB
from torch.utils.data.sampler import BatchSampler
from dataloader import sampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation.recall import give_recall
import argparse


def objective(trial, device):
    thresh = trial.suggest_float("thresh", 0.30, 0.7)
    mvr_reg = trial.suggest_float("mvr_reg", 0.10, 0.7)
    # Transforms
    transforms_tr = trsfrm.Compose([must_transform(), trsfrm.RandomResizedCrop(224), trsfrm.RandomHorizontalFlip()])
    transforms_test = trsfrm.Compose([must_transform(), trsfrm.Resize(256), trsfrm.CenterCrop(224)])
    # Dataset
    root_dir = "./data/CUB_200_2011/images"
    cub_train = CUB(root_dir, 'trainval', transforms_tr)
    cub_val = CUB(root_dir, 'test', transforms_test)
    cuda = torch.device('cuda:{}'.format(device))

    # Model definition
    net = bn_inception(64, pretrained=True, is_norm=True, bn_freeze=False)
    net.to(cuda)

    # DataLoader
    numWorkers = 2
    batch_size = 64
    # SAMPLER IMPLEMENTATION

    if True: # balanced train set
        tr_balanced_sampler = sampler.BalancedSampler(cub_train, batch_size=batch_size, images_per_class=16)
        tr_batch_sampler = BatchSampler(tr_balanced_sampler, batch_size=batch_size, drop_last=True)
        tr_dataloader = torch.utils.data.DataLoader(
            cub_train,
            num_workers=numWorkers,
            pin_memory=True,
            batch_sampler=tr_batch_sampler
        )
    else:
        tr_dataloader = DataLoader(cub_train, batch_size=batch_size, shuffle=True, num_workers=numWorkers, pin_memory=True)

    val_dataloader = DataLoader(cub_val, batch_size=batch_size, shuffle=False, num_workers=numWorkers,
                                pin_memory=True)
    # Loss
    no_tr_class = max(cub_train.target) + 1
    emb_dim = 64
    loss_func = MVR_MS_reg(2.0, 50.0, thresh, 0.1, mvr_reg)
    loss_func.to(cuda)
    # Optimizer
    optimizer = torch.optim.Adam([{"params": net.parameters()},
                                  {"params": loss_func.parameters()}], lr=1e-4)
    # Initial
    best_recall = 0

    for epochs in range(80):
        avg_loss = 0
        net.train()
        for img, lbl in tqdm(tr_dataloader):
            optimizer.zero_grad()
            img = img.to(cuda)
            lbl = lbl.to(cuda)
            embeddings = net(img)
            loss = loss_func(embeddings, lbl)
            avg_loss = avg_loss + loss.item()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            Recalls = give_recall(net, val_dataloader, cuda=cuda)
            trial.report(Recalls[0], epochs)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if Recalls[0] > best_recall:
                best_recall = Recalls[0]

    return best_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVR')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU that is used for training.'
                        )
    args = parser.parse_args()
    study = optuna.create_study(study_name="ms-mvr", storage="sqlite:///ms-mvr.db", direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=40,
                                                                   interval_steps=1), load_if_exists=True)
    study.optimize(lambda trial: objective(trial, args.gpu_id), n_trials=80)

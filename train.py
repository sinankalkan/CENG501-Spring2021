import torchvision.transforms as trsfrm
from dataloader.cub_dataset import CUB
from dataloader.trsfrms import must_transform
from dataloader import sampler
from evaluation.recall import give_recall
from loss.mvrloss import MVR_Proxy, MVR_Triplet, MVR_MS, MVR_MS_reg
from model.bn_inception import bn_inception
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler
import torch
import numpy as np
import random
import os
from tqdm import tqdm
import argparse
import logging


## Triplet margin : 0.3478912374083307 - exp1
## Triplet reg : 0.5061600574032541 - exp1
## Triplet margin : 0.2781877469005122 - exp2
## Triplet reg : 0.4919607680052035 - exp2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVR'
                                     )
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU that is used for training.'
                        )
    parser.add_argument("--tnsrbrd_dir", default="./runs", type=str)
    parser.add_argument("--model_save_dir", default="./MVR_MS/exp", type=str)
    parser.add_argument("--batch_size", default=80, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--wdecay", default=5e-3, type=float)
    parser.add_argument("--mvr_reg", default=0.3, type=float)
    parser.add_argument("--bn_freeze", default=False, type=bool)
    parser.add_argument("--emb_dim", default=64, type=int)
    parser.add_argument("--exp_name", default="exp", type=str)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--balanced_sampler_train", default=True, type=bool)
    parser.add_argument("--balanced_sampler_validation", default=False, type=bool)
    parser.add_argument("--loss", default="ms_reg",type=str)
    parser.add_argument("--margin", default=0.28, type=float)
    parser.add_argument("--images_per_class", default=5, type=int)
    parser.add_argument("--ms_thresh", default=0.6, type=float)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    # seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Tensorboard
    tnsrbrd_dir = args.tnsrbrd_dir
    writer = SummaryWriter(tnsrbrd_dir)
    # model save
    model_save_dir = args.model_save_dir
    if os.path.exists(model_save_dir) != True:
        os.makedirs(model_save_dir)
    # log

    log_dir = os.path.join("./log", args.exp_name)
    if os.path.exists(log_dir) != True:
        os.makedirs(log_dir)

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_dir + "/records.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(
        f"Learning Rate {args.lr}, Weight_decay {args.wdecay}, batch_size {args.batch_size}, emb_dim {args.emb_dim}, patience {args.patience}, mvr_reg = {args.mvr_reg}")

    # Transforms
    transforms_tr = trsfrm.Compose([must_transform(), trsfrm.RandomResizedCrop(224), trsfrm.RandomHorizontalFlip()])
    transforms_test = trsfrm.Compose([must_transform(), trsfrm.Resize(256), trsfrm.CenterCrop(224)])
    # Dataset
    root_dir = "data/CUB_200_2011/images"
    cub_train = CUB(root_dir, 'trainval', transforms_tr)
    cub_val = CUB(root_dir, 'test', transforms_test)

    cuda = torch.device('cuda:{}'.format(args.gpu_id))

    # Model definition
    net = bn_inception(64, pretrained=True, is_norm=True, bn_freeze=args.bn_freeze)
    net.to(cuda)

    # DataLoader
    numWorkers = 4
    batch_size = args.batch_size
    # SAMPLER IMPLEMENTATION
    if args.balanced_sampler_train:
        tr_balanced_sampler = sampler.BalancedSampler(cub_train, batch_size=batch_size, images_per_class=args.images_per_class)
        tr_batch_sampler = BatchSampler(tr_balanced_sampler, batch_size=batch_size, drop_last=True)
        tr_dataloader = torch.utils.data.DataLoader(
            cub_train,
            num_workers=numWorkers,
            pin_memory=True,
            batch_sampler=tr_batch_sampler
        )
    else:
        tr_dataloader = DataLoader(cub_train, batch_size=batch_size, shuffle=True, num_workers=numWorkers, pin_memory=True)
    if args.balanced_sampler_validation:
        val_balanced_sampler = sampler.BalancedSampler(cub_val, batch_size=batch_size, images_per_class = 8)
        val_batch_sampler = BatchSampler(val_balanced_sampler, batch_size = batch_size, drop_last = True)
        val_dataloader = torch.utils.data.DataLoader(
            cub_val,
            num_workers = numWorkers,
            pin_memory = True,
            batch_sampler = val_batch_sampler
        )
    else:
        val_dataloader = DataLoader(cub_val, batch_size=batch_size, shuffle=False, num_workers=numWorkers,
                                    pin_memory=True)
    # Loss
    no_tr_class = max(cub_train.target) + 1
    emb_dim = args.emb_dim

    if args.loss == "triplet":
        loss_func = MVR_Triplet(margin=args.margin, reg=args.mvr_reg)
    elif args.loss == "proxy":
        loss_func = MVR_Proxy(reg=args.mvr_reg, no_class=no_tr_class, embedding_dimension=emb_dim)
    elif args.loss == "ms":
        loss_func = MVR_MS(2.0, 50.0,  0.33121341100189616, 0.1)
    elif args.loss == "ms_reg":
        loss_func = MVR_MS_reg(2.0, 50.0, 0.6, 0.1, args.margin) # 0.5872421417546484)
    loss_func.to(cuda)
    # Optimizer
    if args.wdecay <= 0.0:
        optimizer = torch.optim.Adam([{"params": net.parameters()},
                                      {"params": loss_func.parameters()}], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([{"params": net.parameters(), "weight_decay": args.wdecay},
                                   {"params": loss_func.parameters()}], lr=args.lr)
    # Initial
    best_recall = 0
    patience = 0
    patience_level = args.patience
    epoch_counter = 1
    total_iter_train = int(len(cub_train) / batch_size)

    while patience < patience_level:
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
        avg_loss = avg_loss / total_iter_train
        writer.add_scalar("Training/avg_loss", avg_loss, epoch_counter)
        net.eval()
        with torch.no_grad():
            Recalls = give_recall(net, val_dataloader, cuda=cuda)
            writer.add_scalar("Accuracy/Recall 1", Recalls[0], epoch_counter)
            writer.add_scalar("Accuracy/Recall 2", Recalls[1], epoch_counter)
            writer.add_scalar("Accuracy/Recall 4", Recalls[2], epoch_counter)
            writer.add_scalar("Accuracy/Recall 8", Recalls[3], epoch_counter)
            if Recalls[0] > best_recall:
                best_recall = Recalls[0]
                torch.save(net.state_dict(), os.path.join(model_save_dir, "best.pth"))
                patience = 0
            else:
                patience += 1
        epoch_counter += 1

    logging.info("Best Recall : {}".format(best_recall))

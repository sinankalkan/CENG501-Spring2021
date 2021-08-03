'''
Training the model and evaluating it using downstream classifiers
depending on the used dataset.
'''
import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs import get_datasets
from projection_head import NonLinearProjection
from evaluate import encode_train_set, train_clf, test
from models import *
from cosine_annealing import CosineAnnealingWithLinearRampLR

parser = argparse.ArgumentParser(description='PyTorch self-supervised representation Learning.')
parser.add_argument('--base-lr', default=0.48, type=float, help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='dataset',
                    choices=['cifar10', 'imagenet'])
parser.add_argument('--temperature', type=float, default=0.15, help='InfoNCE temperature')
parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
parser.add_argument("--num-epochs", type=int, default=100, help='Number of training epochs')
parser.add_argument("--cosine-anneal", action='store_true', help="Use cosine annealing on the learning rate")
parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
parser.add_argument("--test-freq", type=int, default=10, help='Frequency to fit a linear clf with L-BFGS for testing'
                                                              'Not appropriate for large datasets. Set 0 to avoid '
                                                              'classifier only training here.')
parser.add_argument("--filename", type=str, default='ckpt.pth', help='Output file name')
args = parser.parse_args()
args.lr = args.base_lr * (args.batch_size / 256)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using the {device}:\n")
best_acc = 0
start_epoch = 0
clf = None

print('==> Preparing data..')
trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                         pin_memory=True)
clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

# Model
print('==> Building model..')
##############################################################
# Encoder
##############################################################
net = ResNet50(stem=stem)
net = net.to(device)

##############################################################
# Projection head with cosine softmax loss
##############################################################
critic = NonLinearProjection(net.representation_dim, temperature=args.temperature).to(device)

if device == 'cuda':
    repr_dim = net.representation_dim
    net = torch.nn.DataParallel(net)
    net.representation_dim = repr_dim
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
encoder_optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6,
                              momentum=args.momentum)
if args.cosine_anneal:
    scheduler = CosineAnnealingWithLinearRampLR(encoder_optimizer, args.num_epochs)


# Training function
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    train_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        loss = criterion(raw_scores, pseudotargets)
        loss.backward()
        encoder_optimizer.step()

        train_loss += loss.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))


###############################################
# contrastive prior:
# We run an inference epoch where we extract instance features by fixing all but BN layers
# of the randomly initialized network, then we directly assign these features to
# classification weights as an initialization
###############################################
def prior_train():
    print('Running inference epoch')
    for name, child in (net.named_children()):
        if name.find('BatchNorm') != -1:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    net.train()
    critic.train()
    train_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, (inputs, _, _) in t:
        x1, x2 = inputs
        x1, x2 = x1.to(device), x2.to(device)
        encoder_optimizer.zero_grad()
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        loss = criterion(raw_scores, pseudotargets)
        loss.backward()
        encoder_optimizer.step()

        train_loss += loss.item()

        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))
    print("Done inference epoch!")
    return raw_scores


# Train
for epoch in range(start_epoch, start_epoch + args.num_epochs):
    init_clf_weights = prior_train()
    # train(epoch)
    if (args.test_freq > 0) and (epoch % args.test_freq == (args.test_freq - 1)):
        print("Encoding...")
        X, y = encode_train_set(clftrainloader, device, net)
        print("Done encoding!")
        clf = train_clf(X, y, net.representation_dim, num_classes, device, init_clf_weights, reg_weight=1e-5)
        acc = test(testloader, device, net, clf)
        if acc > best_acc:
            best_acc = acc
    if args.cosine_anneal:
        scheduler.step()

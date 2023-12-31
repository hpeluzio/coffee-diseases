# -*- coding: utf-8 -*-
'''

Coffee diseases

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import os
import sys 
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

# parsers
parser = argparse.ArgumentParser(description='PyTorch Coffee deseases Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--use_scheduler', action='store_true', help='Scheduler')
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--size', default="224")
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
usewandb = bool(args.nowandb)
# sys.exit()

if usewandb:
    print('Using wandb...')
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="coffee-leaf-diseases-custom-7-classes",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = bool(~args.noamp)
aug = args.noaug

n_epochs = args.n_epochs
batch_size = args.batch_size
use_early_stopping = args.use_early_stopping
use_scheduler = args.use_scheduler
early_stopping_patience = args.patience
num_classes = args.num_classes

print('n_epochs: ', n_epochs)
print('learning rate: ', args.lr)
print('use_scheduler: ', args.use_scheduler)
print('use_early_stopping: ', use_early_stopping)
print('early_stopping_patience: ', early_stopping_patience)
print('num_classes: ', num_classes)
print('batch_size: ', batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize


# Calculated mean: tensor([0.6410, 0.6595, 0.5589])
# Calculated std: tensor([0.2477, 0.2294, 0.3135])

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.Resize(size=[size, size]),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

transform_validation = transforms.Compose([
    transforms.Resize(size=[size, size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

transform_test = transforms.Compose([
    transforms.Resize(size=[size, size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []

        classes = os.listdir(self.root)
        for class_name in classes:
            class_path = os.path.join(self.root, class_name)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                for image_name in images:
                    image_path = os.path.join(class_path, image_name)
                    self.image_paths.append((image_path, int(class_name.split('_')[0])))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Create custom datasets
dataset_folder = r'./dataset/'

train_folder = os.path.join(dataset_folder, 'CUSTOM_COFFEE_LEAF_DATASET_SPLITTED/train')
validation_folder = os.path.join(dataset_folder, 'CUSTOM_COFFEE_LEAF_DATASET_SPLITTED/validation')
test_folder = os.path.join(dataset_folder, 'CUSTOM_COFFEE_LEAF_DATASET_SPLITTED/test')

train_dataset = CustomDataset(root=train_folder, transform=transform_train)
validation_dataset = CustomDataset(root=validation_folder, transform=transform_validation)
test_dataset = CustomDataset(root=test_folder, transform=transform_test)

train_dataset_size = len(train_dataset)
validation_dataset_size = len(validation_dataset)
test_dataset_size = len(test_dataset)

print(f"Number of images in train dataset: {train_dataset_size}")
print(f"Number of images in validation dataset: {validation_dataset_size}")
print(f"Number of images in test dataset: {test_dataset_size}")

# Create data loaders
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validationloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('0_Healthy', '1_CLR_Rust', '2_CLS_Cercospora', '3_PLS_Phoma', '4_CLM_Leaf_Miner', '5_RSM_Red_Spider_Mite', '6_SM_Sooty_Molds')

# Model factory..
print('==> Building model..')

if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = torchvision.models.resnet34(weights='DEFAULT')
    num_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.ReLU(),
        nn.Linear(256, num_classes),
        nn.Softmax(dim=1)
    )
elif args.net=='res50':
    net = ResNet50(num_classes=num_classes)
elif args.net=='res50-torchvision':
    net = torchvision.models.resnet50(weights=None)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_classes)

    # net.fc = nn.Sequential(
    #     nn.Linear(num_features, num_classes),
    #     nn.Softmax(dim=1)
    # )
elif args.net=='res50-torchvision-pretrained':
    net = torchvision.models.resnet50(weights='DEFAULT')
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_classes)

    # net.fc = nn.Sequential(
    #     nn.Linear(num_features, num_classes),
    #     nn.Softmax(dim=1)
    # )
elif args.net=='densenet121':
    net = torchvision.models.densenet121(weights='DEFAULT')
    net.classifier = nn.Linear(1024, num_classes)
elif args.net=='efficientnetb0':
    net = torchvision.models.efficientnet_b0(weights='DEFAULT')
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True), 
        nn.Linear(in_features=1280, out_features=num_classes, bias=True),
    )
elif args.net=='efficientnetb4':
    net = torchvision.models.efficientnet_b4(weights='DEFAULT')
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True), 
        nn.Linear(in_features=1792, out_features=num_classes, bias=True),
        # nn.Softmax(dim=1) 
    )
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # You can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=3)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = num_classes
    )
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
elif args.net=="vit":
    net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=args.num_classes,
                downscaling_factors=(2,2,2,1))

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/coffee-leaf-diseases-custom-7-classes-{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def validation(epoch):
    global best_acc
    net.eval()
    validation_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validationloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validationloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (validation_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print(f'Saving... Accuracy; {(acc):.5f}')
        state = {
            "model": net.state_dict(), 
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/coffee-leaf-diseases-custom-7-classes-'+args.net+'-{}-ckpt.t7'.format(args.patch))
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {validation_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return validation_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
net.cuda()
epochs_since_improvement = 0

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = validation(epoch)
    
    if use_scheduler:
        scheduler.step() # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    # print(list_loss)

    # Early stopping check and update best_acc
    if use_early_stopping == False:
        if acc > best_acc:
            best_acc = acc
    elif use_early_stopping == True:
        if acc > best_acc:
            best_acc = acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch} as validation accuracy has not improved for {early_stopping_patience} epochs.')
            break  # Stop training

print(f'Best validation accuracy: {best_acc}')

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
    

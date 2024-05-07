import os
import sys
import torch
import argparse
import numpy as np
from model import networks
from time import perf_counter
from datasets import ImageDataset
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--epoch_count', type=int, default=1, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='Data', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss:GAN(G(X))')
parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--cuda_device', type=str, default="cuda", help='cuda device to train on')
parser.add_argument('--mode', type=str, default="head", help='training mode')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--test', action='store_true', help="whether is in test phase")
parser.add_argument('--path', type=str, default="./output", help='Path to save model to')
parser.add_argument('--cutmode', type=str, default="cut", help="cut mode")
opt = parser.parse_args()
opt.size = (256, 256)
print(opt)
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

model = networks.defineG(opt.input_nc, opt.output_nc, "cuda:0")
model.load_state_dict(torch.load("/home/aiRobots/output/hand/netG_400.pth"))

transforms_ = [
        transforms.RandomResizedCrop(
            (opt.size[0], opt.size[1]),
            scale = (0.88, 1.0),
            interpolation = transforms.InterpolationMode.BICUBIC,
            antialias=True),
        transforms.Lambda(lambda img: img / 127.5 - 1)
    ]

dataset = ImageDataset(opt.dataroot, transforms_=transforms_, mode = opt.mode)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
epoch = 0
with tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}', unit='batch') as pbar:
    pbar.set_postfix(**{'loss_cycle': 0.0})
    for i, batch in enumerate(dataloader):
        input = Variable(batch["A"].type(torch.Tensor).to("cuda:0"))
        output = model(input)
        output = transforms.Resize((84, 84))(output)
        output = output.cpu()
        save_image(output, f"test/output_{i}.png")
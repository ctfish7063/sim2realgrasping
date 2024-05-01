#!/usr/bin/python3
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm
import torch
import argparse
from model import CycleGAN, CUT
from model import networks
from datasets import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

if __name__=='__main__':
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
    opt.isTrain = not opt.test
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # augmentor = CycleGAN(opt)
    augmentor = CycleGAN(opt)
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
    schedulers = [networks.get_scheduler(o, opt) for o in augmentor.optimizers]
    losses_names = augmentor.losses_names
    visual_names = augmentor.visual_names
    # Loss plot
    writer = SummaryWriter(f'./runs/CycleGAN_{opt.mode}')
    step = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}', unit='batch') as pbar:
            pbar.set_postfix(**{'loss_cycle': 0.0})
            for i, batch in enumerate(dataloader):
                if (epoch == opt.epoch_count and i == 0):
                    augmentor.data_dependent_initialize(batch)
                augmentor.set_input(batch)
                step += 1
                augmentor.optimize()
                loss = augmentor.get_losses()
                assert(len(loss) == len(losses_names))
                for idx, name in enumerate(losses_names):
                    writer.add_scalar(name, loss[idx], step)
                pbar.set_postfix(**{'loss_NCE': loss[-2].detach().cpu().numpy()})
                pbar.update(1)
        vis = augmentor.sample_visual()
        for idx, name in enumerate(visual_names):
            writer.add_image(name, vis[idx][0] * 0.5 + 0.5, epoch)
        # Save models checkpoints
        augmentor.save(epoch, path = f"./output/{opt.mode}")
        # Update learning rates
        for s in schedulers:
            s.step()
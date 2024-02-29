#!/usr/bin/python3
import sys
import torch
import argparse
from model import Augmentor
from datasets import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./Data/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    opt.size = (256, 256)
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    augmentor = Augmentor(opt)
    transforms_ = [transforms.Resize((opt.size[0], opt.size[1]), transforms.InterpolationMode.BICUBIC), transforms.Lambda(lambda img: img / 127.5 - 1)]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    # Loss plot
    writer = SummaryWriter('./runs/CycleGAN')
    step = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        sys.stdout.write(f'\repoch {epoch}/{opt.n_epochs}')
        for i, batch in enumerate(dataloader):
            step += 1
            loss_G, loss_identity, loss_G_GAN, loss_G_cycle, loss_D, real, rSim, sim, sReal = augmentor.step(batch['A'], batch['B'])
            writer.add_scalar('Loss_G',loss_G, step)
            writer.add_scalar('Loss_G_identity',loss_identity, step)
            writer.add_scalar('Loss_G_GAN',loss_G_GAN, step)
            writer.add_scalar('Loss_G_cycle',loss_G_cycle, step)
            writer.add_scalar('Loss_D',loss_D, step)
            writer.add_image('real', real * 0.5 + 0.5, epoch)
            writer.add_image('2Sim', rSim * 0.5 + 0.5, epoch)
            writer.add_image('Sim', sim * 0.5 + 0.5, epoch)
            writer.add_image('2Real', sReal * 0.5 + 0.5, epoch)
        # Update learning rates
        augmentor.lr_step()
        # Save models checkpoints
        augmentor.save(epoch + 1)
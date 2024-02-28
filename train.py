#!/usr/bin/python3
import sys
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
from utils import ReplayBuffer, ReplayMemory, LambdaLR, weights_init_normal
from datasets import ImageDataset
import matplotlib.pyplot as plt

class WGANLoss(torch.nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        if target:
            return -torch.mean(input)
        return torch.mean(input)

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

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)
    if opt.epoch != 0:
        print(f"Loading epoch {opt.epoch}")
        netG_A2B.load_state_dict(torch.load(f'./output/netG_A2B_cycle_{opt.epoch}.pth'))
        netG_B2A.load_state_dict(torch.load(f'./output/netG_B2A_cycle_{opt.epoch}.pth'))
        netD_B.load_state_dict(torch.load(f'./output/netD__B_cycle_{opt.epoch}.pth'))
        netD_A.load_state_dict(torch.load(f'./output/netD_A_cycle_{opt.epoch}.pth'))
    else:
        print("Starting from scratch")
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)
    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    # criterion_GAN = WGANLoss().cuda()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    # optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    # lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size[0], opt.size[1])
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False).unsqueeze(1)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False).unsqueeze(1)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [transforms.Resize((opt.size[0], opt.size[1]), transforms.InterpolationMode.BICUBIC), transforms.Lambda(lambda img: img / 127.5 - 1)]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    # Loss plot
    writer = SummaryWriter('./runs/CycleGAN')
    ###################################

    ###### Training ######
    step = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        sys.stdout.write(f'\repoch {epoch}/{opt.n_epochs}')
        for i, batch in enumerate(dataloader):
            step += 1
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B) # B -> B
            loss_identity_B = criterion_identity(same_B, real_B) * opt.lambda_B * opt.lambda_identity
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A) # A -> A
            loss_identity_A = criterion_identity(same_A, real_A) * opt.lambda_A * opt.lambda_identity

            # GAN loss
            fake_B = netG_A2B(real_A) # A -> B
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B) # B -> A
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * opt.lambda_A

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * opt.lambda_B

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            for param in netD_A.parameters():
                param.requires_grad = False
            for param in netD_B.parameters():
                param.requires_grad = False
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            for param in netD_A.parameters():
                param.requires_grad = True
            for param in netD_B.parameters():
                param.requires_grad = True
            ###################################

            ###### Discriminator A ######

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # optimizer_D_A.zero_grad()
            # loss_D_A.backward()

            # optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            # optimizer_D_B.zero_grad()
            # loss_D_B.backward()

            # optimizer_D_B.step()
            optimizer_D.zero_grad()
            loss_D_A.backward()
            loss_D_B.backward()
            optimizer_D.step()
            ###################################

            writer.add_scalar('Loss_G',loss_G,step)
            writer.add_scalar('Loss_G_identity',loss_identity_A + loss_identity_B,step)
            writer.add_scalar('Loss_G_GAN',loss_GAN_A2B + loss_GAN_B2A,step)
            writer.add_scalar('Loss_G_cycle',loss_cycle_ABA + loss_cycle_BAB,step)
            writer.add_scalar('Loss_D',loss_D_A + loss_D_B,step)
            writer.add_image('real', real_A[0] * 0.5 + 0.5, epoch)
            writer.add_image('2Sim', fake_B[0] * 0.5 + 0.5, epoch)
            writer.add_image('Sim', real_B[0] * 0.5 + 0.5, epoch)
            writer.add_image('2Real', fake_A[0] * 0.5 + 0.5, epoch)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        # lr_scheduler_D_A.step()
        # lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), './output/netG_cycle_A2B_%d.pth'%(epoch+1))
        torch.save(netG_B2A.state_dict(), './output/netG_cycle_B2A_%d.pth'%(epoch+1))
        torch.save(netD_A.state_dict(), './output/netD_cycle_A_%d.pth'%(epoch+1))
        torch.save(netD_B.state_dict(), './output/netD__cycle_B_%d.pth'%(epoch+1))
    ###################################
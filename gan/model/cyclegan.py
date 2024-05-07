import os
import torch
import itertools
from .buffer import ReplayBuffer
from . import networks
from .losses import GANLoss

class CycleGAN():
    def __init__(self, opt):
        os.environ['CUDA_VISIBLE_DEVICE']='0,1'
        self.lambda_identity = opt.lambda_identity
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.device = torch.device(opt.cuda_device if opt.cuda and torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        self.netG_A2B = networks.defineG(opt.input_nc, opt.output_nc, self.device)
        self.netG_B2A = networks.defineG(opt.output_nc, opt.input_nc, self.device)
        self.losses_names = ['loss_G', 'loss_GAN', 'loss_Idt', 'loss_Cycle', 'loss_D_A', 'loss_D_B']
        self.visual_names = ['fake_A', 'fake_B', 'real_A', 'real_B']
        if opt.isTrain:
            self.netD_A = networks.defineD(opt.input_nc, self.device)
            self.netD_B = networks.defineD(opt.output_nc, self.device)
            if opt.epoch != 0:
                print(f"Loading epoch {opt.epoch}")
                self.load(opt.path, opt.epoch)
            self.criterion_GAN = GANLoss('lsgan').to(self.device)
            self.criterion_cycle = torch.nn.L1Loss()
            self.criterion_identity = torch.nn.L1Loss()
            # Optimizers & LR schedulers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            # Inputs & targets memory allocation
            self.fake_A_buffer = ReplayBuffer()
            self.fake_B_buffer = ReplayBuffer()
        else:
            self.netG_A2B.eval()
            self.netG_B2A.eval()

    def data_dependent_initialize(self, data):
        pass

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A2B(self.real_A)  # G_A(A)
        self.recovered_A = self.netG_B2A(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B2A(self.real_B)  # G_B(B)
        self.recovered_B = self.netG_A2B(self.fake_A)   # G_A(G_B(B))

    def compute_G_loss(self):
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = self.netG_A2B(self.real_B) # B -> B
        loss_identity_B = self.criterion_identity(same_B, self.real_B) * self.lambda_B * self.lambda_identity
        # G_B2A(A) should equal A if real A is fed
        same_A = self.netG_B2A(self.real_A) # A -> A
        loss_identity_A = self.criterion_identity(same_A, self.real_A) * self.lambda_A * self.lambda_identity
        loss_GAN_A2B = self.criterion_GAN(self.netD_B(self.fake_B), True)
        loss_GAN_B2A = self.criterion_GAN(self.netD_A(self.fake_A), True)
        loss_cycle_ABA = self.criterion_cycle(self.recovered_A, self.real_A) * self.lambda_A
        loss_cycle_BAB = self.criterion_cycle(self.recovered_B, self.real_B) * self.lambda_B
        self.loss_Idt = loss_identity_A + loss_identity_B
        self.loss_GAN = loss_GAN_A2B + loss_GAN_B2A
        self.loss_Cycle = loss_cycle_ABA + loss_cycle_BAB
        self.loss_G = self.loss_Idt + self.loss_GAN + self.loss_Cycle
        self.loss_G.backward()

    def compute_D_loss(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def compute_D_A_loss(self):
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A)
        self.loss_D_A = self.compute_D_loss(self.netD_A, self.real_A, fake_A)

    def compute_D_B_loss(self):
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B)
        self.loss_D_B = self.compute_D_loss(self.netD_B, self.real_B, fake_B)

    def optimize(self):
        self.forward()
        for param in self.netD_A.parameters():
            param.requires_grad = False
        for param in self.netD_B.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()
        self.compute_G_loss()
        self.optimizer_G.step()
        for param in self.netD_A.parameters():
            param.requires_grad = True
        for param in self.netD_B.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        self.compute_D_A_loss()
        self.compute_D_B_loss()
        self.optimizer_D.step()
    
    def get_losses(self):
        return [self.loss_G, self.loss_GAN, self.loss_Idt, self.loss_Cycle, self.loss_D_A, self.loss_D_B]
    
    def sample_visual(self):
        return [self.fake_A[0], self.fake_B[0], self.real_A[0], self.real_B[0]]
    
    def load(self, path, epoch):
        self.netG_A2B.load_state_dict(torch.load(f'{path}/netG_A2B_cycle_{epoch}.pth'))
        self.netG_B2A.load_state_dict(torch.load(f'{path}/netG_B2A_cycle_{epoch}.pth'))
        if self.isTrain:
            self.netD_B.load_state_dict(torch.load(f'{path}/netD_B_cycle_{epoch}.pth'))
            self.netD_A.load_state_dict(torch.load(f'{path}/netD_A_cycle_{epoch}.pth'))

    def save(self, epoch, path = "./output"):
        if (os.path.exists(path) == False):
            os.mkdir(path)
        torch.save(self.netG_A2B.state_dict(), f'{path}/netG_A2B_cycle_{epoch}.pth')
        torch.save(self.netG_B2A.state_dict(), f'{path}/netG_B2A_cycle_{epoch}.pth')
        torch.save(self.netD_A.state_dict(), f'{path}/netD_A_cycle_{epoch}.pth')
        torch.save(self.netD_B.state_dict(), f'{path}/netD_B_cycle_{epoch}.pth')
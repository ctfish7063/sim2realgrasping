import itertools
import torch
from gan import Generator, Discriminator
from torch.autograd import Variable
from utils import ReplayBuffer

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Augmentor():
    def __init__(self, opt, path = "./output"):
        self.lambda_identity = opt.lambda_identity
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.netG_A2B = Generator(opt.input_nc, opt.output_nc)
        self.netG_B2A = Generator(opt.output_nc, opt.input_nc)
        self.netD_A = Discriminator(opt.input_nc)
        self.netD_B = Discriminator(opt.output_nc)
        if opt.epoch != 0:
            print(f"Loading epoch {opt.epoch}")
            self.netG_A2B.load_state_dict(torch.load(f'{path}/netG_A2B_cycle_{opt.epoch}.pth'))
            self.netG_B2A.load_state_dict(torch.load(f'{path}/netG_B2A_cycle_{opt.epoch}.pth'))
            self.netD_B.load_state_dict(torch.load(f'{path}/netD__B_cycle_{opt.epoch}.pth'))
            self.netD_A.load_state_dict(torch.load(f'{path}/netD_A_cycle_{opt.epoch}.pth'))
        else:
            print("Starting from scratch")
            self.netG_A2B.apply(weights_init_normal)
            self.netG_B2A.apply(weights_init_normal)
            self.netD_A.apply(weights_init_normal)
            self.netD_B.apply(weights_init_normal)
        if opt.cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        # Optimizers & LR schedulers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        self.input_A = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
        self.input_B = Tensor(opt.batchSize, opt.output_nc, opt.size[0], opt.size[1])
        self.target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False).unsqueeze(1)
        self.target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False).unsqueeze(1)
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def step(self, batch_A, batch_B):
        real_A = Variable(self.input_A.copy_(batch_A))
        real_B = Variable(self.input_B.copy_(batch_B))
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = self.netG_A2B(real_B) # B -> B
        loss_identity_B = self.criterion_identity(same_B, real_B) * self.lambda_B * self.lambda_identity
        # G_B2A(A) should equal A if real A is fed
        same_A = self.netG_B2A(real_A) # A -> A
        loss_identity_A = self.criterion_identity(same_A, real_A) * self.lambda_A * self.lambda_identity
        # GAN loss
        fake_B = self.netG_A2B(real_A) # A -> B
        pred_fake = self.netD_B(fake_B)
        loss_GAN_A2B = self.criterion_GAN(pred_fake, self.target_real)
        fake_A = self.netG_B2A(real_B) # B -> A
        pred_fake = self.netD_A(fake_A)
        loss_GAN_B2A = self.criterion_GAN(pred_fake, self.target_real)
        # Cycle loss
        recovered_A = self.netG_B2A(fake_B)
        loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * self.lambda_A
        recovered_B = self.netG_A2B(fake_A)
        loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * self.lambda_B
        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        #No need to calculate gradients for discriminators
        for param in self.netD_A.parameters():
            param.requires_grad = False
        for param in self.netD_B.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        for param in self.netD_A.parameters():
            param.requires_grad = True
        for param in self.netD_B.parameters():
            param.requires_grad = True
        ###### Discriminator A ######
        # Real loss
        pred_real = self.netD_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, self.target_real)
        # Fake loss
        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
        pred_fake = self.netD_A(fake_A.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)
        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        ###### Discriminator B ######
        # Real loss
        pred_real = self.netD_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, self.target_real)
        # Fake loss
        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
        pred_fake = self.netD_B(fake_B.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        self.optimizer_D.zero_grad()
        loss_D_A.backward()
        loss_D_B.backward()
        self.optimizer_D.step()
        return loss_G, loss_identity_A + loss_identity_B, loss_GAN_A2B + loss_GAN_B2A, loss_cycle_ABA + loss_cycle_BAB, loss_D_A + loss_D_B, real_A[0], fake_B[0], real_B[0], fake_A[0]
    
    def lr_step(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()
    
    def save(self, epoch, path = "./output"):
        torch.save(self.netG_A2B.state_dict(), f'{path}/netG_A2B_cycle_{epoch}.pth')
        torch.save(self.netG_B2A.state_dict(), f'{path}/netG_B2A_cycle_{epoch}.pth')
        torch.save(self.netD_A.state_dict(), f'{path}/netD_A_cycle_{epoch}.pth')
        torch.save(self.netD_B.state_dict(), f'{path}/netD_B_cycle_{epoch}.pth')

if __name__ == "__main__":
    #sanity check
    opt = type('obj', (object,), {})()
    opt.epoch = 0
    opt.n_epochs = 200
    opt.batchSize = 1
    opt.dataroot = './Data/'
    opt.lr = 0.0002
    opt.decay_epoch = 100
    opt.input_nc = 3
    opt.output_nc = 3
    opt.lambda_A = 10.0
    opt.lambda_B = 10.0
    opt.lambda_identity = 0.5
    opt.cuda = True
    opt.size = (256, 256)
    augmentor = Augmentor(opt)
    print(augmentor)
    print(augmentor.step(torch.rand(1,3,256,256), torch.rand(1,3,256,256)))
    augmentor.lr_step()
    augmentor.save(1, path = "./tmp/")
    print("Sanity check passed")
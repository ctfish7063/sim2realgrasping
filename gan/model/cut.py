import os
import torch
import numpy as np
import itertools
from .networks import defineG, defineF, defineD
from .losses import PatchNCELoss, GANLoss

class CUT():
    def __init__(self, opt):
        os.environ['CUDA_VISIBLE_DEVICE']='0,1'
        opt.lambda_GAN=1.0
        opt.nce_T=0.07
        opt.num_patches=256
        opt.flip_equivariance=False
        opt.nce_includes_all_negatives_from_minibatch=False
        if opt.cutmode == "cut":
            opt.nce_idt = True
            opt.lambda_NCE=1.0
        elif opt.cutmode == "fastcut":
            opt.nce_idt = False
            opt.lambda_NCE=10.0
            opt.flip_equivariance=True
            opt.n_epochs=150
            opt.n_epochs_decay=50
        else:
            raise ValueError(opt.cutmode)
        self.opt = opt
        self.optimizers = []
        self.lambda_identity = opt.lambda_identity
        self.lambda_GAN = opt.lambda_GAN
        self.lambda_NCE = opt.lambda_NCE
        self.device = torch.device(opt.cuda_device if opt.cuda and torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        self.netG = defineG(opt.input_nc, opt.output_nc, self.device)
        self.netF = defineF(256, self.device)
        self.losses_names = ['G_GAN', 'D_real', 'D_fake', 'NCE', 'NCE_Y']
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'fake_B_idt']
        if opt.isTrain:
            self.netD = defineD(opt.output_nc, self.device)
            self.nce_layers = [0, 4, 8, 12, 16]
            if opt.epoch != 0:
                print(f"Loading epoch {opt.epoch}")
                self.load(opt.epoch, opt.path)
            self.criterionGAN = GANLoss("lsgan").to(self.device)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        else:
            self.netG.eval()
            self.netF.eval()
    
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_F)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def optimize(self):
        # forward
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def get_losses(self):
        return [self.loss_G, self.loss_D_real, self.loss_D_fake, self.loss_NCE, self.loss_NCE_Y]

    def sample_visual(self):
        return [self.real_A, self.real_B, self.fake_B, self.idt_B]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN
        self.loss_NCE = self.compute_NCE_loss(self.real_A, self.fake_B)
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.compute_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def compute_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def load(self, path, epoch):
        self.netG.load_state_dict(torch.load(path + f"/netG_{epoch}.pth"))
        self.netF.load_state_dict(torch.load(path + f"/netF_{epoch}.pth"))
        if self.opt.isTrain:
            self.netD.load_state_dict(torch.load(path + f"/netD_{epoch}.pth"))

    def save(self, epoch, path = "./output"):
        if (os.path.exists(path) == False):
            os.mkdir(path)
        torch.save(self.netF.state_dict(), f'{path}/netF_{epoch}.pth')
        torch.save(self.netG.state_dict(), f'{path}/netG_{epoch}.pth')
        torch.save(self.netD.state_dict(), f'{path}/netD_{epoch}.pth')

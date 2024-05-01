import os
import sys
import torch
import argparse
import numpy as np
from model import Generator
from time import perf_counter
from datasets import ImageDataset
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./Data/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_cycle_A2B_200.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_cycle_B2A_200.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
opt.size = (256, 256)
print(opt)
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size[0], opt.size[1])

# Dataset loader
transforms_ = [transforms.Resize((opt.size[0], opt.size[1]), transforms.InterpolationMode.BICUBIC), transforms.Lambda(lambda img: img / 127.5 - 1)]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='train'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
# Create output dirs if they don't exist
path = os.path.join(opt.dataroot, 'generated')
secs = []
for i, batch in enumerate(dataloader):
    # Set model input
    start_t = perf_counter()
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    label = batch['B_paths']
    file_nameB = label[0].split('/')[-1]
    label = batch['A_paths']
    file_nameA = label[0].split('/')[-1]
    # Generate output
    fake_A = netG_B2A(real_B)
    fake_B = netG_A2B(real_A)
    fake_A = fake_A * 0.5 + 0.5
    fake_B = fake_B * 0.5 + 0.5
    # Save image files
    save_image(fake_A, os.path.join(path, "real", file_nameB), normalize=True)
    save_image(fake_B, os.path.join(path, "sim", file_nameA), normalize=True)
    end_t = perf_counter()
    secs.append(end_t-start_t)
    sys.stdout.write('\rGenerated images %04d of %04d in %04f sec' % (i+1, len(dataloader), end_t-start_t))
sys.stdout.write('\n')
mean = np.mean(secs)
std = np.std(secs)
print('Average time taken to generate an image: ', mean)
print('Standard deviation of time taken to generate an image: ', std)
secs = [i for i in secs if abs(i - mean) < 2 * std]
plt.boxplot(secs, showfliers=False, labels=[])
plt.title('Generate time for each image')
plt.show()
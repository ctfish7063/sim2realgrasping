import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',rate=1.0):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob(os.path.join(root, '%s/real' % mode) + '/*.*'))
        self.files_B = sorted(glob(os.path.join(root, '%s/sim' % mode) + '/*.*'))
        self.files_A = random.sample(self.files_A, int(len(self.files_A) * rate))
        self.files_B = random.sample(self.files_B, int(len(self.files_B) * rate))

    def __getitem__(self, index):
        A_Path = self.files_A[index % len(self.files_A)]
        B_Path = self.files_B[index % len(self.files_B)]
        item_A = read_image(A_Path, mode = ImageReadMode.RGB).to(dtype = torch.float32)
        item_A = self.transform(item_A)
        item_B = read_image(B_Path, mode = ImageReadMode.RGB).to(dtype = torch.float32)
        item_B = self.transform(item_B)
        return {'A': item_A, 'B': item_B, 'A_paths': A_Path, 'B_paths': B_Path}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
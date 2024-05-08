import os
import torch
import random
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='head',rate=1.0):
        self.__transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A_head = sorted(glob(os.path.join(root, 'real/', mode) + '/*.jpg'))
        # self.files_A_hand = sorted(glob(os.path.join(root, 'real/hand') + '/*.jpg'))
        self.files_B_head = sorted(glob(os.path.join(root, 'sim/', mode) + '/*.jpg'))
        # self.files_B_hand = sorted(glob(os.path.join(root, 'sim/hand') + '/*.jpg'))
        # self.files_A = list(zip(self.files_A_head, self.files_A_hand))
        # self.files_A = random.sample(self.files_A, int(len(self.files_A) * rate))
        self.files_A = random.sample(self.files_A_head, int(len(self.files_A_head) * rate))
        # self.files_B = list(zip(self.files_B_head, self.files_B_hand))
        self.files_B = random.sample(self.files_B_head, int(len(self.files_B_head) * rate))
        # self.files_B = random.sample(self.files_B_head, int(len(self.files_B) * rate))
        
    def transform(self, imglist):
        flip = random.random() > 0.5
        for i in range(len(imglist)):
            img = imglist[i]
            img = self.__transform(img)
            imglist[i] = img
        return imglist
    
    def __getitem__(self, index):
        A_Path_head = self.files_A[index % len(self.files_A)]
        # A_Path_hand = self.files_A[index % len(self.files_A)][1]
        B_Path_head = self.files_B[index % len(self.files_B)]
        # B_Path_hand = self.files_B[index % len(self.files_B)][1]
        # A_Path_head = self.files_A[index % len(self.files_A)]
        # B_Path_head = self.files_B[index % len(self.files_B)]
        item_A_head = read_image(A_Path_head, mode = ImageReadMode.RGB).to(dtype = torch.float32)
        # item_A_hand = read_image(A_Path_hand, mode = ImageReadMode.RGB).to(dtype = torch.float32)
        item_B_head = read_image(B_Path_head, mode = ImageReadMode.RGB).to(dtype = torch.float32)
        # item_B_hand = read_image(B_Path_hand, mode = ImageReadMode.RGB).to(dtype = torch.float32)
        # items = [item_A_head, item_A_hand, item_B_head, item_B_hand]
        # items = [item_A_head, item_B_head]
        items = [item_A_head, item_B_head]
        items = self.transform(items)
        item_A = items[0]
        item_B = items[1]
        return {'A': item_A, 'B': item_B, 'A_paths': A_Path_head, 'B_paths': B_Path_head}

    def __len__(self):
        return max(len(self.files_A_head), len(self.files_B_head))
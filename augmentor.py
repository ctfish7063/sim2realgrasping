import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .gan.model import networks

class Augmentor(nn.Module):
    class opt:
        def __init__(self):
            self.size = (256, 256)
            self.isTrain = False
            self.epoch = 0
            self.epoch_count = 1
            self.batchSize = 1
            self.dataroot = 'Data'
            self.lr = 0.0002
            self.n_epochs = 200
            self.n_epochs_decay = 200
            self.input_nc = 3
            self.output_nc = 3
            self.lambda_A = 10.0
            self.lambda_B = 10.0
            self.lambda_GAN = 1.0
            self.lambda_identity = 0.5
            self.cuda = True
            self.cuda_device = "cuda:0"
            self.mode = "hand"
            self.n_cpu = 1
            self.test = True
            self.path = "./output"
            self.cutmode = "cut"

    def __init__(self, model_path, device):
        super(Augmentor, self).__init__()
        self.opt = Augmentor.opt()
        self.device = device
        self.model = networks.defineG(self.opt.input_nc, self.opt.output_nc, self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.transform = [
            transforms.Resize(
                (self.opt.size[0], self.opt.size[1]),
                interpolation = transforms.InterpolationMode.BICUBIC,
                antialias=True),
            transforms.Lambda(lambda img: img / 127.5 - 1)]
        self.transform = transforms.Compose(self.transform)
        self.model.eval()

    def augment(self, img, **kwargs):
        with torch.no_grad():
            img = img.to(self.device)
            ret = self.model(img, **kwargs)
            ret = transforms.Resize((84, 84))(ret)
            ret = ret.squeeze().cpu().numpy()
            return ret
        
    def foward(self, img, **kwargs):
        img = self.transform(img)
        return self.augment(img, **kwargs)
    
if __name__ == "__main__":
    print("sanity test")
    augmentor = Augmentor("/home/aiRobots/output/hand/netG_400.pth", "cuda:0")
    x = torch.randn(1, 3, 84, 84)
    y = augmentor.foward(x, layers = [0, 4, 8, 12, 16], encode_only = True)
    for feat in y:
        print(feat.size())
import torch
from time import time
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import Generator
import matplotlib.pyplot as plt
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CycleGAN:
    def __init__(self, opt):
        self.opt = opt
        self.netG_B2A = Generator(opt.input_nc, opt.output_nc)
        if opt.cuda:
            self.netG_B2A.cuda()
        self.netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
        self.netG_B2A.eval()
        self.Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
        self.transforms_ = [transforms.ToTensor(), transforms.Resize((opt.size[0], opt.size[1]), transforms.InterpolationMode.BICUBIC), transforms.Lambda(lambda img: img / 127.5 - 1)]
        self.transforms_ = transforms.Compose(self.transforms_)
        self.listener = rospy.Subscriber("headimg", Image, self.callback)
        self.bridge = CvBridge()
        self.inbuffer = []
        self.genbuffer = []
    def generate(self, img):
        img = self.transforms_(img)
        img = img.unsqueeze(0)
        input = Variable(self.input_A.copy_(img))
        return self.netG_B2A(input).data * 0.5 + 0.5
    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data,"rgba8")
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            self.inbuffer.append(img)
        except CvBridgeError as e:
            print(e)
    def show(self, img):
        tensor_image = self.generate(img)
        img = tensor_image[0].permute(1, 2, 0).cpu().numpy()
        cv2.imshow('generated', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)
    def run(self):
        cv2.namedWindow('generated', cv2.WINDOW_AUTOSIZE)
        while not rospy.is_shutdown():
            try:
                if len(self.inbuffer) > 0:
                    self.show(self.inbuffer.pop(0))
                plt.show()
            except KeyboardInterrupt:
                break
if __name__ == '__main__':
    class Opt:
        def __init__(self):
            self.input_nc = 3
            self.output_nc = 3
            self.cuda = True
            self.generator_B2A = "output/netG_cycle_B2A_200.pth"
            self.batchSize = 1
            self.size = (256, 256)
    opt = Opt()
    rospy.init_node('calvin_cyclegan')
    cyclegan = CycleGAN(opt)
    cyclegan.run()
    rospy.spin()
import torch
import pathlib
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from Actor import Actor

features = {}


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)


path_img = 'runs/detect/exp/crops/person/0000014.jpg'
img = Image.open(path_img)
w,h = img.size
x, y = w/2, h/2
img = img.resize((416, 416))
img_arr = np.array(img)
#print(img_arr.shape)
transform = transforms.ToTensor()
img_tr = transform(img_arr)
img_tr = torch.unsqueeze(img_tr, 0)
#matplotlib.use('TkAgg')
#plt.imshow(img_tr.view(416, 416, 3))
#plt.show()

if __name__ == '__main__':
    #log_dir = pathlib.Path.cwd() / "tensorboard1_logs"
    #writer = SummaryWriter(log_dir)
    model = darknet53(1)
    #print(model)

    #def activation_hook(inst, inp, out):
     #   writer.add_histogram(repr(inst), out)

    #model.residual_block5.register_forward_hook(activation_hook)
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook


    model.global_avg_pool.register_forward_hook(get_features('person'))
    z = model(img_tr)

    bbox = torch.Tensor([x, y, w, h])
    bbox = torch.unsqueeze(bbox, 0)
    bbox = torch.unsqueeze(bbox, 2)
    bbox = torch.unsqueeze(bbox, 3)
    #print(y)
    #print(features['person'])
    #print(y.shape)
    state = torch.cat([bbox, features['person']], 1)
    state = torch.squeeze(state, 0)
    state = torch.reshape(state, [1, 1, 1028])
    print(state)
    state_size = state.shape[2]
    action_size = 4

    #initialize the agent
    action = Actor(state_size, action_size)
    print(action)
    y = action(state)
    print(y)


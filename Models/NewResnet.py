import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, channels):
       super(ResnetBlock, self).__init__() 

       self.channels = channels
       self.RNB1 = self._layer_struct(self.channels)
       self.RNB2 = self._layer_struct(self.channels)

    def _layer_struct(self, ch):
        layr = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch)
            ,nn.ReLU()
        ) 
        return layr

    def forward(self, x):
        RNB1_out = self.RNB1(x)
        RNB2_out = self.RNB1(RNB1_out)
        return RNB2_out

class BuildBlock(nn.Module):

    def __init__(self, in_ch, out_ch, Is_ResNetBlock_req):
        super(BuildBlock, self).__init__()

        self.base_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_ch)
            ,nn.ReLU()
        ) 

        self.RNB_flg  = Is_ResNetBlock_req

        if self.RNB_flg  == 1:
            self.RNB = ResnetBlock(out_ch)

    def forward(self, x):
        out1 = self.base_layer(x)
        if self.RNB_flg  == 1:
            R1 = self.RNB(out1)
            out = out1 + R1
            return out
        return out1


class newResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(newResNet, self).__init__()
        #self.in_planes = 64

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
            ,nn.BatchNorm2d(64)
            ,nn.ReLU()
        ) 

        self.layer1 = self._make_layer(block, 64, 128, 1)
        self.layer2 = self._make_layer(block, 128, 256, 0)
        self.layer3 = self._make_layer(block, 256, 512, 1)
        self.MP4 = nn.MaxPool2d(4, 4)
        self.FC = nn.Linear(512,num_classes)
        #self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self,block,in_ch, out_ch, Is_ResNetBlock_req_flg):
        layers = []
        layers.append(block(in_ch, out_ch, Is_ResNetBlock_req_flg))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.preplayer(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.MP4(out4)
        
        out6 = out5.view(out5.size(0), -1)
        #print(out6.shape)
        out7 = self.FC(out6)
        out = F.softmax(out7, dim=-1)
        #print(out.shape)
        return out

def CustomResNet():
    return newResNet(BuildBlock)
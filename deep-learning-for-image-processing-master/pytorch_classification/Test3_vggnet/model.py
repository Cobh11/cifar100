# -*- coding:UTF-8 -*-
import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),               #dorpout防止过拟合
            nn.Linear(512*7*7, 2048),    #原论文是4096
            nn.ReLU(True),
            nn.Dropout(p=0.5),           #全连接层1与2之间加一个百分之五十的dropout
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        self.classifier2 = nn.Sequential(nn.Linear(num_classes, 2048),
            nn.ReLU(True),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048,512*7*7),
            nn.Dropout(p = 0.5)
            #nn.MaxUnPool2d
        )
        self.cnn2 = nn.Sequential(
                                 # nn.MaxUnpool2d(kernel_size = 2, stride = 2),
                                  nn.Conv2d(512,512,kernel_size = 3, padding = 1),
                                  nn.Conv2d(512,512,kernel_size = 3, padding = 1),
                                  nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                                  #nn.MaxUnpool2d(kernel_size = 2, stride = 2),
                                  nn.Conv2d(512,512,kernel_size = 3, padding = 1),
                                  nn.Conv2d(512,256,kernel_size = 3, padding = 1),
                                  nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                                  #nn.MaxUnpool2d(kernel_size = 2, stride = 2),
                                  nn.Conv2d(256,256,kernel_size = 3, padding = 1),
                                  nn.Conv2d(256,128,kernel_size = 3, padding = 1),
                                  #nn.MaxUnpool2d(kernel_size = 2, stride = 2),
                                  nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                                  nn.Conv2d(128,64,kernel_size = 3, padding = 1),
                                  nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                                  #nn.MaxUnpool2d(kernel_size = 2, stride = 2),
                                  nn.Conv2d(64,3,kernel_size = 3, padding =1)
                                  )

        if init_weights:
            self._initialize_weights()     #如果传入的init_weights=True,用下面定义的函数初始化

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)   #start_dim:从哪个维度开始展屏，因为第零个维度是batch维度，所以从第一个维度开始
        # N x 512*7*7
        x1 = self.classifier1(x)
        x2 = self.classifier2(x1)
        x2 = x2.view(-1,512,7,7)
        a = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        x2 = a(x2)
        x2 = self.cnn2(x2) 
        return x1,x2

    def _initialize_weights(self):
        for m in self.modules():        #遍历每一个层
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:               #如果采用了偏置的话，偏置置为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#%%生成了nn.Sequential
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)    #星号表示通过非关键字参数传入进去


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#%%实例化vgg16
def vgg(model_name="vgg16", **kwargs):   #**kwargs:可变长度的字典变量
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model

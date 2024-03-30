"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M'], # VGG7 - CIFAR10
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512],
    'C' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'F' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG_CIFAR(nn.Module):

    def __init__(self, features, num_class=10, activation='relu', cfg=[512, 'M']):
        super().__init__()
        if activation == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activate = nn.Tanh(inplace=True)
        elif activation == 'sigmoid':
            self.activate = nn.Sigmoid(inplace=True)
        elif activation == 'leaky_relu':
            self.activate = nn.LeakyReLU(inplace=True)
        
        self.feature_extraction = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-2], 512),
            self.activate,
            nn.Dropout(),
            nn.Linear(512, 512),
            self.activate,
            nn.Dropout(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        output = self.feature_extraction(x)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False, activation='relu'):
    if activation == 'relu':
        activate = nn.ReLU(inplace=True)
    elif activation == 'tanh':
        activate = nn.Tanh(inplace=True)
    elif activation == 'sigmoid':
        activate = nn.Sigmoid(inplace=True)
    elif activation == 'leaky_relu':
        activate = nn.LeakyReLU(inplace=True)
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [activate]
        input_channel = l

    return nn.Sequential(*layers)

def vgg7_cifar10_bn(num_classes=10, activation='relu'):
    return VGG_CIFAR(make_layers(cfg['A'], batch_norm=True, activation=activation), num_class=num_classes, activation=activation, cfg=cfg['A'])

def vgg9_cifar10_bn(num_classes=10, activation='relu'):
    return VGG_CIFAR(make_layers(cfg['B'], batch_norm=True, activation=activation), num_class=num_classes, activation=activation, cfg=cfg['B'])

def vgg11_cifar10_bn(num_classes=10, activation='relu'):
    return VGG_CIFAR(make_layers(cfg['C'], batch_norm=True, activation=activation), num_class=num_classes, activation=activation, cfg=cfg['C'])

def vgg13_cifar10_bn(num_classes=10, activation='relu'):
    return VGG_CIFAR(make_layers(cfg['D'], batch_norm=True, activation=activation), num_class=num_classes, activation=activation, cfg=cfg['D'])

def vgg16_cifar10_bn(num_classes=10, activation='relu'):
    return VGG_CIFAR(make_layers(cfg['E'], batch_norm=True, activation=activation), num_class=num_classes, activation=activation, cfg=cfg['E'])

def vgg19_cifar10_bn(num_classes=10, activation='relu'):
    return VGG_CIFAR(make_layers(cfg['F'], batch_norm=True, activation=activation), num_class=num_classes, activation=activation, cfg=cfg['F'])

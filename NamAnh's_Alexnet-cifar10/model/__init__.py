from .alexnet import *
from .minimal_alexnet import *
from .chaunet import ChauNet


def get_model(
    name='alexnet',
    num_classes=10,
    activation='relu',
    dropout=True
):
    if name == 'alexnet':
        return AlexNet(num_classes=num_classes, activation=activation, dropout=dropout)
    elif name == 'minimal_alexnet':
        return Minimal_Alexnet(num_classes=num_classes, activation=activation, dropout=dropout)
    elif name == 'chaunet':
        return ChauNet(num_classes=num_classes, activation=activation, dropout=dropout)
    else:
        raise ValueError("Only support options: alexnet")
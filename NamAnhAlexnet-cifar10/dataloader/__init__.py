from .cifar10 import *


def get_dataloader(
    data='cifar10', 
    data_augmentation='basic',
    batch_size=128,
    num_workers=4
    ):
    if data == 'cifar10':
        return get_cifar10(
            data_augmentation=data_augmentation,
            batch_size=batch_size,
            num_workers=num_workers
        )
    elif data == 'mnist':   
        pass
    elif data =='cifar100':
        pass
    else:
        raise ValueError("Only support options: cifar10")
    
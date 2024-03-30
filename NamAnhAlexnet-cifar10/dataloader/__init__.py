from .cifar10 import *
from .cifar100 import *

def get_dataloader(
    data_name='cifar10', 
    data_augmentation='basic',
    batch_size=128,
    num_workers=4
    ):
    if data_name == 'cifar10':
        return get_cifar10(
            data_augmentation=data_augmentation,
            batch_size=batch_size,
            num_workers=num_workers
        )
    elif data_name == 'mnist':   
        pass
    elif data_name =='cifar100':
        return get_cifar100(
            data_augmentation=data_augmentation,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        raise ValueError("Only support options: cifar10")
    
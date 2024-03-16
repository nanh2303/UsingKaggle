from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms


def get_cifar10(
    data_augmentation='basic',
    batch_size=128,
    num_workers=4
    ):
    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    if data_augmentation == 'basic':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    elif data_augmentation == 'sota': # State of The Art
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
            transforms.RandomRotation(degrees=30),  # Randomly rotate the image up to 30 degrees
            transforms.ColorJitter(brightness=0.2),  # Randomly adjust brightness with a factor of 0.2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    elif data_augmentation == 'minimal': # State of The Art
        transform_train = transforms.Compose([
            transforms.Resize(40),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
            transforms.RandomRotation(degrees=30),  # Randomly rotate the image up to 30 degrees
            transforms.ColorJitter(brightness=0.2),  # Randomly adjust brightness with a factor of 0.2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
        transform_test = transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    else:
        raise ValueError("Only support options: basic or sota")
    
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform_train
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform_test
    )

    training_data, validation_data = random_split(training_data, lengths=(0.9, 0.1))

    train_dataloader = DataLoader(
        training_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_dataloader = DataLoader(
        validation_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader, test_dataloader, classes
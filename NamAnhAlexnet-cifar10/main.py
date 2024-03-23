import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import yaml
import argparse
import wandb

from utils import *
from dataloader import *
from model import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
best_acc = 0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', type=str, help='path to YAML config file')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.Loader)

wandb.init(
    project=cfg['data_name'], 
    name=f"{cfg['model_name']}-bs={cfg['batch_size']}-lr={cfg['learning_rate']}-wd={cfg['weight_decay']}-aug={cfg['data_augmentation_type']}-act={cfg['activation']}", 
)
log_dict = {}
test_dict = {}

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, classes = get_dataloader(
    data=cfg['data_name'], 
    data_augmentation=cfg['data_augmentation_type'], 
    batch_size=cfg['batch_size'], 
    num_workers=cfg['num_workers']
)

################################
#### 2. BUILD THE NEURAL NETWORK
################################
model = get_model(
    name=cfg['model_name'], 
    num_classes=len(classes),
    activation=cfg['activation'],
    dropout=cfg['dropout']
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), 
    lr=cfg['learning_rate'],
    weight_decay=cfg['weight_decay']
)

################################
#### 3.b Training 
################################
if __name__ == '__main__':
    for epoch in range(cfg['epochs']):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_one_epoch(
            dataloader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device, 
            batch_size=cfg['batch_size'],
            log_dict=log_dict
        )
        
        best_acc = validation_one_epoch(
            epoch=epoch, 
            dataloader=val_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            device=device,
            current_time=current_time,
            best_acc=best_acc,
            log_dict=log_dict
        )
        
        wandb.log(log_dict)
        
    test_one_epoch(
        dataloader=test_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        device=device,
        current_time=current_time,
        log_dict=test_dict
    )
    
    wandb.log(test_dict)

    ################################
    #### 3.c Testing on each class
    ################################
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
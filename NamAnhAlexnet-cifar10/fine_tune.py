import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import yaml
import argparse
import wandb
import pprint

from utils import *
from dataloader import *
from model import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
# 0. SETUP CONFIGURATION
################################
best_acc = 0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', type=str, help='path to YAML config file')
parser.add_argument('--pretrain_path', type=str, help='path to pretrain checkpoint')
args = parser.parse_args()

args.pretrain_path = os.path.join('.', 'checkpoint', '20240330_163653')
assert os.path.exists(args.pretrain_path), 'Error: Pretrain path does not exist!!!'

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.Loader)
    pprint.pprint(cfg)

EPOCHS = cfg['trainer']['epochs']

wandb_name = 'FINE_TUNE'
wandb_name += '_MODEL'
for key, value in sorted(cfg['model'].items()):
    wandb_name += f'_{key[:2]}={value}'
wandb_name += '_OPT'
for key, value in sorted(cfg['optimizer'].items()):
    wandb_name += f'_{key[:2]}={value}'
wandb_name += '_DATA'
for key, value in sorted(cfg['dataloader'].items()):
    wandb_name += f'_{key[:2]}={value}'
wandb_name += '_TRAINER'
for key, value in sorted(cfg['trainer'].items()):
    wandb_name += f'_{key[:2]}={value}'
    
logging_name = cfg['dataloader']['data_name'] + '_' + wandb_name  + '_' + current_time

wandb.init(
    project=cfg['dataloader']['data_name'],
    name=wandb_name,
)
log_dict = {}
test_dict = {}

################################
# 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, classes = get_dataloader(
    **cfg['dataloader'])
try:
    num_classes = len(classes)
except:
    num_classes = classes
################################
# 2. BUILD THE NEURAL NETWORK
################################
checkpoint = torch.load(os.path.join(args.pretrain_path, 'ckpt_best.pth'))
model = get_model(
    **cfg['model'],
    num_classes=100,  # Hardcode CIFAR100
)
model.load_state_dict(checkpoint['model'])
model.classifier = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(512, num_classes)
)

if cfg['trainer']['freeze_extraction']:
    freeze_layer = 'feature_extraction'
    for name, param in model.named_parameters():
        if freeze_layer in name:
            param.requires_grad = False
        else: param.requires_grad = True
        
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

################################
# 3.a OPTIMIZING MODEL PARAMETERS
################################
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    **cfg['optimizer']
)

################################
# 3.b Training
################################
if __name__ == '__main__':
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_one_epoch(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            log_dict=log_dict
        )

        best_acc = validation_one_epoch(
            epoch=epoch, 
            dataloader=val_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            device=device,
            logging_name=logging_name,
            best_acc=best_acc,
            log_dict=log_dict
        )
        
        wandb.log(log_dict)
        
    test_one_epoch(
        dataloader=test_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        device=device,
        logging_name=logging_name,
        log_dict=test_dict
    )
    wandb.log(test_dict)

    ################################
    # 3.c Testing on each class
    ################################
    if isinstance(classes, list):
        class_correct = list(0. for _ in range(num_classes))
        class_total = list(0. for _ in range(num_classes))
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
        table_data = [{classes[i]: 100 * class_correct[i]}
                      for i in range(num_classes)]
        table = wandb.Table(data=table_data, columns=["Class", "Accuracy"])
        wandb.log({"Test Accuracy per Class": table})

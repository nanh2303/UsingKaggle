import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
################################
#### 1. BUILD THE DATASET
################################
transform = transforms.Compose([
    transforms.Resize(256),         # (32x32)   -> (256x256)
    transforms.CenterCrop(227),     # (256x256) -> (227x227)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize: (0, 255) -> (0, 1)
]) # Data augmentation => [mean - 1*std, mean + 1*std] -> 67% du lieu nam trong khoang gia tri nay
                    #     [mean - 2*std, mean + 2*std] -> 95% du lieu nam trong khoang gia tri nay
                    #     [mean - 3*std, mean + 3*std] -> 99% du lieu nam trong khoang gia tri nay

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

training_data, validation_data = random_split(training_data, lengths=(0.9, 0.1))

batch_size = 128  
EPOCHS = 200
hidden_size = 512
learning_rate = 1e-3
lambda_l2 = 0.0
num_classes = 10
   
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
################################
#### 2. BUILD THE NEURAL NETWORK
################################
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),            # Tranh overfitting
            
            nn.Linear(9216, 4096),
            nn.ReLU(inplace = True),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extraction(x)  # (6x6x256) 
        x = self.flatten(x)             # 9216
        logit = self.classifier(x)      # num_classes, ^y

        return logit

model = AlexNet(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

################################
#### 3.b Training 
################################
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss, acc = 0, 0
    for batch, (X, y) in enumerate(dataloader): # for X, y in dataloader
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)                 # model.forward(X)
        loss = loss_fn(y_pred, y)         # 
        train_loss += loss.item()
        acc += (y_pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()                   # Back-propagation
        optimizer.step()                  # update weight: w_{t+1} = w_t - lr * grad(w_t)
        optimizer.zero_grad()             # grad(w_t) = 0

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    train_loss /= num_batches
    acc /= size
    print(f"Validation: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    
def validation_loop(epoch, dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    global best_acc
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, acc = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            acc += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'loss': val_loss,
            'epoch': epoch
        }
        if not os.path.isdir(f'checkpoint/{current_time}'):
            os.makedirs(f'checkpoint/{current_time}')
        torch.save(state, f'./checkpoint/{current_time}/ckpt_best.pth')
        best_acc = acc

    val_loss /= num_batches
    acc /= size
    print(f"Validation: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{current_time}/ckpt_best.pth')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        validation_loop(t, val_dataloader, model, loss_fn)
    test_loop(test_dataloader, model, loss_fn)

    print("Done!")

    ################################
    #### 3.d Testing on each class
    ################################
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
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

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
import torch
import os
from .utils import *


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, log_dict):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch, (X, y) in enumerate(dataloader): # for X, y in dataloader
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)                 # model.forward(X)
        loss = loss_fn(y_pred, y)         # 
        train_loss += loss.item()
        correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()                   # Back-propagation
        optimizer.step()                  # update weight: w_{t+1} = w_t - lr * grad(w_t)
        optimizer.zero_grad()             # grad(w_t) = 0

        total += y.size(0)
        train_loss_mean = train_loss/(batch+1)
        acc = 100.*correct/total
        progress_bar(batch, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss_mean, acc, correct, total))
    
    log_dict['train/loss'] = train_loss_mean
    log_dict['train/acc'] = acc


def validation_one_epoch(epoch, dataloader, model, loss_fn, device, logging_name, best_acc, log_dict):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    val_loss, correct, total = 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            total += y.size(0)
            val_loss_mean = val_loss/(batch+1)
            acc = 100.*correct/total
            progress_bar(batch, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (val_loss_mean, acc, correct, total))
        
    log_dict['val/loss'] = val_loss_mean
    log_dict['val/acc'] = acc
            
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'loss': val_loss_mean,
            'epoch': epoch
        }
        if not os.path.isdir(f'checkpoint/{logging_name}'):
            os.makedirs(f'checkpoint/{logging_name}')
        torch.save(state, f'./checkpoint/{logging_name}/ckpt_best.pth')
        best_acc = acc

    return best_acc
    
def test_one_epoch(dataloader, model, loss_fn, device, logging_name, log_dict):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{logging_name}/ckpt_best.pth')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    test_loss, correct, total = 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            total += y.size(0)
            test_loss_mean = test_loss/(batch+1)
            acc = 100.*correct/total
            progress_bar(batch, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss_mean, acc, correct, total))
        
    log_dict['test/loss'] = test_loss_mean
    log_dict['test/acc'] = acc
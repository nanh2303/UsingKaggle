import torch
import os


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, batch_size, log_dict):
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
    print(f"Training: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    log_dict['train/loss'] = train_loss
    log_dict['train/acc'] = 100*acc


def validation_one_epoch(epoch, dataloader, model, loss_fn, device, current_time, best_acc, log_dict):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
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
    log_dict['val/loss'] = val_loss
    log_dict['val/acc'] = 100*acc
    
    return best_acc
    
def test_one_epoch(dataloader, model, loss_fn, device, current_time, log_dict):
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
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    log_dict['test/loss'] = test_loss
    log_dict['test/acc'] = 100*correct
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:07:04 2020

@author: 11028434
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
 
    import torch
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        inputs, labels = data.to(device), target.to(device)
        
        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
        
        # Predict
        y_pred = model(inputs)
        
        # Calculate loss
        loss = criterion(y_pred, labels)
        train_loss += train_loss
        
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update pbar-tqdm
        #pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        _, predicted = torch.max(y_pred.data, 1)
        correct += (predicted == labels).sum().item()
        processed += len(inputs)
        
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss)
    return train_acc,train_losses

def test(model, device, test_loader):
    import torch
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            output = model(images)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, labels).sum().item()
            
            _, predicted = torch.max(output.data, 1)
            #total += labels.size(0)
            correct += (predicted == labels).sum().item()       
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))
    
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return test_acc,test_loss

def get_misclassified_images(model, device, test_loader):
    import torch
    model.eval()
    test_loss = 0
    correct = 0
    flg =0
    with torch.no_grad():
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            output = model(images)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, labels).sum().item()
            
            _, predicted = torch.max(output.data, 1)
            #total += labels.size(0)
            correct += (predicted == labels).sum().item()

            mis_c = predicted != labels
            if flg == 0:
                misc_im = data[mis_c]
                misc_tr = labels[mis_c]
                misc_pred = predicted[mis_c]
                flg =1
            else:  
                misc_im = torch.cat((data[mis_c],misc_im))
                misc_tr = torch.cat((labels[mis_c],misc_tr))
                misc_pred = torch.cat((predicted[mis_c],misc_pred)) 
    return misc_im, misc_tr, misc_pred


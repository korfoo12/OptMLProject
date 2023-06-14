import torch
import torch.nn as nn
import torch.nn.init as I

import random
import numpy as np

def batch_accuracy(outputs, labels):
    """
    compute accuracy on an output batch
    
    Args:
        outputs : output batch
        labels : ground truth labels batch
    """
    with torch.no_grad():
        _, predictions = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

def fit(net, dataloader, epochs, opt, lr, **kwargs):
    """
    train a model
    
    Args:
        net : model to train
        dataloader : loads data used for training
        epochs: number of epochs to run training
        opt : optimizer
        lr : learning rate
        kwargs : optional keyword arguments to pass to the optimizer
        
    Returns:
        history : train/validation loss and accuracy at each epoch
    """
    optimizer = opt(net.parameters(), lr=lr, **kwargs)
    history = []
    
    net.train()
    for _ in range(epochs):
        losses = []
        accs = []
        for images,labels in dataloader:

            optimizer.zero_grad()

            outputs = net(images)

            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            acc = batch_accuracy(outputs, labels)
            
            losses.append(loss)
            accs.append(acc)

        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        history.append((avg_loss, avg_acc))

    return history

def evaluate(net, dataloader):
    """
    evaluate (loss and accuracy) a trained model on some data
    
    Args:
        net : trained model
        dataloader : loads the data to evaluate the model on
    """
    losses = []
    accs = []
    net.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            outputs = net(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            acc = batch_accuracy(outputs, labels)
            losses.append(loss)
            accs.append(acc)
    avg_loss = torch.stack(losses).mean().item()
    avg_acc = torch.stack(accs).mean().item()
    return (avg_loss, avg_acc)

def init_layer(l : nn.Module):
    """
    initialize a layer
    For Convolutional and Fully Connected layers weights, Kaiming Normal initialization in fan-in mode is used
    Batch Norm weights are set to 1
    All biases are set to 0
    
    Args:
        l : layer
    """
    cname = l.__class__.__name__
    if "Conv2d" in cname:
        I.kaiming_normal_(l.weight,mode='fan_in',nonlinearity='relu')
        if l.bias is not None:
            I.torch.nn.init.constant_(l.bias,0.)

    elif 'BatchNorm2d' in cname:
        I.constant_(l.weight, 1.)
        I.constant_(l.bias, 0.)

    elif 'Linear' in cname:
        I.kaiming_normal_(l.weight,mode='fan_in',nonlinearity='relu')
        if l.bias is not None:
            I.torch.nn.init.constant_(l.bias,0.)

    elif 'Sequential' in cname:
        for c in l.children():
            init_layer(c)

def seed_generators():
    """
    seed generators and enforce deterministic cuda for reproducibility
    """
    torch.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)
    torch.backends.cudnn.deterministic = True


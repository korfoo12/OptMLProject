import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions.dirichlet import Dirichlet
from itertools import groupby
import random

class ListDataset(Dataset):
    """
    Dataset wrapper around a list
    """
    def __init__(self,l):
        self.l = l.copy()
        
    def __len__(self):
        return len(self.l)
    
    def __getitem__(self,i):
        return self.l[i]
    
    def __str__(self):
        return self.l.__str__()
    
    def __repr__(self):
        return self.l.__repr__()
    
class DeviceDataLoader(DataLoader):
        """
        Dataloader which loads samples on the device
        """
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)

def partition_by_class(dataset: Dataset):
    """
    Partition a dataset by class
    
    Args:
        dataset
        
    Returns:
        partition : dict with entries (i,list of samples from the dataset with label i)
    """
    key = lambda x: x[1]
    return {k:list(vs) for k,vs in groupby(sorted(dataset,key=key), key)}

def split(partition, proportions):
    """
    split datasets into subdatasets to be used by clients based on sampled propotions
    
    Args:
        partition : dataset formated as a dict with entries (i,list of samples from the dataset with label i)
        proportions : sampled proportions which dictacte how many of data samples of each class to assign to each client dataset
        
    Returns:
        subdatasets : list of datasets obtained from splitting
    """
    nodes = [list() for _ in proportions[0]]
    
    # iterate class and add a random nb of samples to each node
    for (_,vs),p in zip(partition.items(),proportions):
        random.shuffle(vs)
        
        len_per_client = p * len(vs)
        indices = torch.cat((torch.zeros(1),len_per_client.cumsum(0).round()),0).long()
        
        for i,(start,stop) in enumerate(zip(indices[:-1],indices[1:])):
            nodes[i] += vs[start:stop]

    return [ListDataset(node) for node in nodes]

def generate_proportions(num_clients, num_classes, alpha=0.1):
    """
    For each class, generate the proportion of the dataset to allocate to each client
    via Dirichlet(alpha) distribution
    
    Args:
        num_clients : number of clients
        num_classes : number of classes
        alpha : parameter of the Dirichlet distribution, default=0.1
        
    Return:
        proportions
        
    """
    splitter = Dirichlet(torch.ones(num_clients)*alpha)
    return [splitter.sample() for _ in range(num_classes)]

def get_device():
    """
    get the device to use
    """
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    """
    send data to device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


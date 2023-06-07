import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.distributions.dirichlet import Dirichlet
from itertools import groupby
import random

class ListDataset(Dataset):
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
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)

def partition_by_class(dataset: Dataset):
    key = lambda x: x[1]
    return {k:list(vs) for k,vs in groupby(sorted(dataset,key=key), key)}

def split(partition, nb_nodes: int, alpha: float = 1.):
    splitter = Dirichlet(torch.ones(nb_nodes)*alpha)
    nodes = [list() for i in range(nb_nodes)]
    
    # iterate class and add a random nb of samples to each node
    for k,vs in partition.items():
        random.shuffle(vs)
        
        nbs = splitter.sample() * len(vs)
        indices = torch.cat((torch.zeros(1),nbs.cumsum(0).round()),0).long()
        
        for i,(start,stop) in enumerate(zip(indices[:-1],indices[1:])):
            nodes[i] += vs[start:stop]
    return [ListDataset(node) for node in nodes]

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
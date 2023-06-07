import torch.nn as nn
import torch

class Server:    
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        
    def merge(self, clients):
        merged_state_dict = self.net.state_dict()

        for k,_ in merged_state_dict.items():
            if k.endswith(".weight") or k.endswith(".bias"):
                merged_state_dict[k] = 0

        len_dataset = sum([len(c.dataset) for c in clients])
        for c in clients:
            w = len(c.dataset) / len_dataset
            for k, param in c.net.state_dict().items():
                if k.endswith(".weight") or k.endswith(".bias"):
                    merged_state_dict[k] += w * param

        self.net.load_state_dict(merged_state_dict)

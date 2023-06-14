import torch.nn as nn
from lib.train_helper import init_layer

class Server:
    """
    Represents the server which combines parameters issued from the clients' trainings
    
    Args:
        net
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        
    def merge(self, clients):
        """
        merge clients updates according to the Federated Averaging algorithm
        
        args:
            clients: clients containing learned parameters of the current round
        """
        merged_state_dict = self.net.state_dict()

        for k,_ in merged_state_dict.items():
            merged_state_dict[k] = 0

        len_dataset = sum([len(c.dataset) for c in clients])
        for c in clients:
            w = len(c.dataset) / len_dataset
            for k, param in c.net.state_dict().items():
                merged_state_dict[k] += w * param

        self.net.load_state_dict(merged_state_dict)

    def reset_weights(self):
        """
        initialize weights
        """
        for layer in self.net.children():
            init_layer(layer)

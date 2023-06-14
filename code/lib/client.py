from lib.data_helper import DeviceDataLoader
from torch.utils.data import DataLoader
from lib.train_helper import *

class Client:
    """
    Represents a client
    
    Args:
        client_id : unique identifier
        dataset : dataset on which training is done
        net : model used for training
    """
    def __init__(self, client_id, dataset, net):
        self.client_id = client_id
        self.dataset = dataset
        self.net = net
    
    def train(self, device, state_dict, epochs, batch_size, opt, lr, **kwargs):
        """
        train the client's network on the client's dataset for a number of epoch
        
        Args:
            device
            state_dict : parameters of the centralized server's model to use
            epochs
            batch_size
            opt : optimizer
            lr : learning rate
            **kwargs : optional arguments to pass to the optimizer
        """
        dataloader = DeviceDataLoader(DataLoader(self.dataset, batch_size, shuffle=True), device)
        self.net.load_state_dict(state_dict)
        train_history = fit(self.net, dataloader, epochs, opt, lr, **kwargs)
        print(f'client {self.client_id} : Loss = {train_history[-1][0]:.4f}, Accuracy = {train_history[-1][1]:.4f}')
from lib.data_helper import DeviceDataLoader
from torch.utils.data import DataLoader
from lib.train_helper import *

class Client:
    def __init__(self, client_id, dataset, net):
        self.client_id = client_id
        self.dataset = dataset
        self.net = net
    
    def train(self, device, state_dict, epochs, batch_size, opt, lr, **kwargs):
        dataloader = DeviceDataLoader(DataLoader(self.dataset, batch_size, shuffle=True), device)
        self.net.load_state_dict(state_dict)
        train_history = fit(self.net, dataloader, epochs, opt, lr, **kwargs)
        print(f'client {self.client_id} : Loss = {train_history[-1][0]:.4f}, Accuracy = {train_history[-1][1]:.4f}')
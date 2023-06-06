from lib.data_helper import DeviceDataLoader
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
    
    def get_dataset_size(self):
        return len(self.dataset)
    
    def get_client_id(self):
        return self.client_id
    
    def train(self, net, parameters_dict, epochs_per_client, learning_rate, batch_size, device, optim):
        dataloader = DeviceDataLoader(DataLoader(self.dataset, batch_size, shuffle=True), device)
        net.apply_parameters(parameters_dict)
        train_history = net.fit(dataloader, epochs_per_client, learning_rate, opt=optim)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        return net.get_parameters()
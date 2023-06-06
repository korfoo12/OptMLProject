import torch
from torch.utils.data import DataLoader

class FederatedNet(torch.nn.Module):    
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
    
    def forward(self, x_batch):
        out = self.net(x_batch)
        return out
    
    def get_track_layers(self):
        return self.track_layers

    def apply_parameters(self, net_state_dict):
        has_customized_func = hasattr(self.net, 'apply_parameters') and callable(getattr(self.net, 'apply_parameters'))
        if has_customized_func:
            self.net.apply_parameters(net_state_dict)
        else:
            self.net.load_state_dict(net_state_dict)
        
    def merge_parameters(self, params):
        state_dict = {}
        for k,_ in params[0].items():
            state_dict[k] = 0
            
        for w, sd in params:
            for k, v in sd.items():
                state_dict[k] += w * v
        self.net.load_state_dict(state_dict)
        
    def get_parameters(self):
        has_customized_func = hasattr(self.net, 'get_parameters') and callable(getattr(self.net, 'get_parameters'))
        if has_customized_func:
            return self.net.get_parameters()
        else:
            return self.net.state_dict()
    
    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
    
    def _process_batch(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)
    
    def fit(self, dataloader, epochs, lr, opt=torch.optim.SGD):
        optimizer = opt(self.parameters(), lr)
        history = []
        for epoch in range(epochs):
            losses = []
            accs = []
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
                losses.append(loss)
                accs.append(acc)
            avg_loss = torch.stack(losses).mean().item()
            avg_acc = torch.stack(accs).mean().item()
            history.append((avg_loss, avg_acc))
        return history

    def evaluate(self, dataloader):
        losses = []
        accs = []
        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)
    
    def save(self):
        return {
            "state_dict" : self.state_dict(),
            "net_state_dict": self.net.state_dict()
        }
    
    @classmethod
    def load(cls,state_dict,get_net):
        net = get_net()
        net.load_state_dict(state_dict)
        return cls(net)
        
    
import torch
import torch.nn as nn

def batch_accuracy(outputs, labels):
    with torch.no_grad():
        _, predictions = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

def fit(net, dataloader, epochs, opt, lr, **kwargs):
    optimizer = opt(net.parameters(), lr=lr, **kwargs)
    history = []
    
    for _ in range(epochs):
        losses = []
        accs = []
        for images,labels in dataloader:

            outputs = net(images)

            loss = nn.functional.cross_entropy(outputs, labels)
            acc = batch_accuracy(outputs, labels)

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

def evaluate(net, dataloader):
    losses = []
    accs = []
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
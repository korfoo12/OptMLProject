import torch.nn as nn

def model():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64,track_running_stats=False),
        nn.ReLU(),
        
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64,track_running_stats=False),
        nn.ReLU(),
        
        nn.Conv2d(64, 128, 3),
        nn.BatchNorm2d(128,track_running_stats=False),
        nn.ReLU(),
        
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(128, 128, 3),
        nn.BatchNorm2d(128,track_running_stats=False),
        nn.ReLU(),
        
        nn.Conv2d(128, 256, 3),
        nn.BatchNorm2d(256,track_running_stats=False),
        nn.ReLU(),
        
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(256, 256, 3),
        nn.BatchNorm2d(256,track_running_stats=False),
        nn.ReLU(),
        
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 10),
    )
import torch
import torch.nn as nn

class vggWrapper(nn.Module):
    
    def __init__(self,base_net,out_dim):
        super().__init__()
        features = list(base_net.features.children()).copy()
        
        # insert batch norm layers
        for i in range(len(base_net.features))[::-1]:
            if i % 5 == 0 or i % 5 == 2:
                features.insert(i+1,nn.BatchNorm2d(features[i].out_channels))
                
        self.features = nn.Sequential(*features)
        
        self.scale = base_net.avgpool
        self.classifier = nn.Sequential(
            next(base_net.classifier.children()),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096,out_features=out_dim,bias=True)        
        )
        
    def forward(self,x):
        x = self.features(x)
        x = self.scale(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
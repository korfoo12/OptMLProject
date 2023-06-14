import torch
import torch.nn as nn

class VGGWrapper(nn.Module):
    
    def __init__(self,base_net,out_dim):
        super().__init__()
        features = list(base_net.features.children()).copy()
        
        # insert batch norm layers
        inserted = 0
        for i,f in enumerate(base_net.features.children()):
            if "Conv2d" in f.__class__.__name__:
                features.insert(i+1+inserted,nn.BatchNorm2d(f.out_channels,track_running_stats=False))
                inserted +=1
                
        self.features = nn.Sequential(*features)
        
        #self.scale = base_net.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512,out_features=out_dim,bias=True)        
        )
        
    def forward(self,x):
        x = self.features(x)
        #x = self.scale(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
import torch.nn as nn
import torch.nn.functional as F


class DummyCNN(nn.Module):
    def __init__(self, config, convs, mlp, in_size_, out_size_):
        super(DummyCNN, self).__init__()
        self.config = config
        
        self.convs = []
        in_size = in_size_[0]
        for out_size in convs:
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, 
                                           kernel_size=self.config["kernel_size"], 
                                           padding=self.config["kernel_size"]//2),
                                nn.MaxPool2d(2),
                                nn.BatchNorm2d(out_size),
                                nn.ReLU())
            self.convs.append(conv)
            in_size = out_size
        self.convs = nn.Sequential(*self.convs)
        
        self.flatten = nn.Flatten()
        
        self.mlp = []
        in_size = convs[-1] * (in_size_[1] // (2**(len(convs)))) ** 2
        for out_size in mlp:
            layer = nn.Sequential(nn.Linear(in_size, out_size),
                                    nn.BatchNorm1d(out_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
            self.mlp.append(layer)
            in_size = out_size
        self.mlp.append(nn.Linear(in_size, out_size_[1]))
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        
        x = self.flatten(x)
        
        for layer in self.mlp:
            x = layer(x)
        
        return x
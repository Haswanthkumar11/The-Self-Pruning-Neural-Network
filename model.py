import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize gates biased towards OFF
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2)

    def forward(self, x):
        # Sharper sigmoid for stronger pruning behavior
        gates = torch.sigmoid(self.gate_scores * 10)
        
        pruned_weight = self.weight * gates
        
        return F.linear(x, pruned_weight, self.bias)


class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = PrunableLinear(32*32*3, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
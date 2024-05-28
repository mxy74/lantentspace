﻿import torch
import torch.nn as nn
import torch.nn.functional as F

class Rob_predictor(nn.Module):
    def __init__(self):
        super(Rob_predictor, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,10)
        self.fc4 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
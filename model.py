import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # Input layer (4 features of the flower) --> Hidden layers
    # --> Output layer (3 classes of flowers)
    def __init__(self, in_features=4, h1=8, h2=8, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x = self.out(x)

        return x


# PICK A MANUAL SEED FOR RANDOMIZATION
torch.manual_seed(42)
model = Model()

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
We use an artificial neural network consisting of a fully connected layer with 4 input features and 15 output features, a second fully connected layer with 15 input features and 20 output features, and an output layer with 20 input features and 3 output features. We use ReLU as our activation function.
'''
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=15)
        self.fc2 = nn.Linear(in_features=15, out_features=20)
        self.output = nn.Linear(in_features=20, out_features=3)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
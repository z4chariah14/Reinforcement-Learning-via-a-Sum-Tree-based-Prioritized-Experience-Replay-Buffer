import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=128):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.Linear3 = nn.Linear(hidden_size, num_outputs)
        self.Relu = nn.ReLU()

    def forward(self, input):
        x = self.linear1(input)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Relu(x)
        x = self.Linear3(x)
        return x

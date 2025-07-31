# models/pixel_mlp32_tanh.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class PixelMLP32Tanh(nn.Module):
    def __init__(self, in_dim=5, out_dim=3, layer_size=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.out = nn.Linear(layer_size, out_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.out(x))          # final tanh keeps range (-1,1)
        return F.normalize(x, dim=-1)        # re-normalise for unit normals

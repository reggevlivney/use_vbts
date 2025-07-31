from pixel_mlp import PixelMLP32Tanh
import torch
import numpy as np
from pixel_dataset import GelSightPixelDataset
# def predict_normal(nn,rgb,pixel):


drgb = np.array([0,0,0]).astype(np.float32)
# pixel = np.array([120,160]).astype(np.float32)
pixel = np.array([20,20]).astype(np.float32)

nn = PixelMLP32Tanh(5,3,64).to('cpu')
nn.load_state_dict(torch.load('pixel_mlp_normals.pth', map_location='cpu',weights_only=True))

imH = 246
imW = 328
imDims = np.array([imW,imH]).astype(np.float32)
pixel_net = 2*(pixel/imDims) - 1
drgb_net = drgb/255

net_input = np.array([np.concatenate([drgb,pixel_net])])

with torch.no_grad():
    chunk = torch.from_numpy(net_input).to('cpu')
    net_output = nn(chunk).cpu().numpy()
print(f'Input: {net_input}')
print(f'Output: {net_output}')

g = GelSightPixelDataset(split=None)
drgbvals, nmapvals = g.get_pixel_values(20,20)

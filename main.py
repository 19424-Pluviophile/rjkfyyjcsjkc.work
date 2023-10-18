import torch
import torch.nn as nn
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        output = x+1
        return output

tudui = Tudui()
x = torch.tensor(3)
output = tudui(x)
print(output)

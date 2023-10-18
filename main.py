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
output = tudui(x)#深度学习中自动调用前向传播forward
print(output)
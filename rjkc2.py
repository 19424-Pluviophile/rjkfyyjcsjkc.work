import torch
import torch.nn as nn
from tifffile.tifffile_geodb import Linear
from torch.nn import Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader
import torchvision
train_data=torchvision.datasets.CIFAR100
test_data=torchvision.datasets.CIFAR100
class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module=nn.Sequential()
        Conv2d(3, 32, 5, padding=2),
        MaxPool2d(2),
        Conv2d(32, 32, 5, padding=2),
        MaxPool2d(2),
        Conv2d(32, 64, 5, padding=2),
        MaxPool2d(2),
        Flatten(),
        Linear(1024, 64),
        Linear(64, 10)

    def forward(self,x):
        output=self.model(x)
        return output
tudui=Tudui()
cross=nn.CrossEntropyLoss()
optim=torch.optim.SGD(params=tudui.parameters(),lr=le-2)
total_loss=0
for epoch in range(20):
    for data in train_loader:
        img,label=data
        output=tudui(img )
        loss_fn=cross(output,label)

        optim.zero_grad()
        loss_fn.backward()
        optim.step()
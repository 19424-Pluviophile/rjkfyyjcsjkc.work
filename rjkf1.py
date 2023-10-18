import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size)->None:
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.sigmoid=torch.nn.Sigmoid()
        self.fc2=nn.Linear(hidden_size,output_size)
    def forward(self,x,w1,w2):
        self.fc1.weight.data=w1
        self.fc1.bias.data=torch.Tensor([0])
        h=self.fc1(x)
        h=self.sigmoid(h)

        self.fc2.weight.data = w2
        self.fc2.bias.data = torch.Tensor([0])
        o = self.fc1(h)
        o = self.sigmoid(o)
        return o
if __name__ =='__main__':
    x=torch.tensor([0.5,0.3])
    w1=torch.tensor([[0.2,0.5],[-0.4,0.6]])
    w2=torch.tensor([[0.1,-0.3],[-0.5,0.8]])

    net=Net(2,2,2)
    output=net(x,w1,w2)
    print('最终输出的结果为：',output)
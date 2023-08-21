from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear,Sigmoid

class Net(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod = Sequential(
            Conv2d(3,32,11,2,4),
            MaxPool2d(4),
            Conv2d(32,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Sigmoid(),
            Flatten(),
            Linear(3136,128),
            Linear(128,2)
        )
    
    def forward(self,x):
        x = self.mod(x)
        return x

    

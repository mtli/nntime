import torch
from torch import nn

from nntime import set_global_sync, time_this, timer_start, timer_end, export_timings
set_global_sync(True)
# import nntime to change the global settings before importing
# other modules that uses nntime


class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 7)

    # tag the functions you want to time
    @time_this()
    def forward(self, x):
        return self.conv(x)
    
class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 4)

    # Don't tag boring functions
    def forward(self, x):
        return self.linear(x)

class Stage3(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(1)

    # tag the functions you want to time
    @time_this()
    def forward(self, x):
        return self.softmax(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1, self.stage2, self.stage3 = Stage1(), Stage2(), Stage3()

    @time_this()
    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        timer_start(self, 'stage2+3')
        x = self.stage2(x)
        x = self.stage3(x)
        timer_end(self, 'stage2+3')
        return x

def main():
    dataset = [torch.randn([2, 3, 7, 7]) for _ in range(10)]
    model = Model()
    for data in dataset:
        y = model(data)
    
    export_timings(model, 'timings.csv')

if __name__ == '__main__':
    main()

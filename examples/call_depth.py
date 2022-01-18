'''
This example shows how to better name the timers and provide a numerical
semantic call depth to better present and organize the results. Note that
this value can be different from the actual call depth, which can be hard
to interpret due to the existence of various wrappers in complex codebase.
'''

import torch
from torch import nn
import torch.nn.functional as F

from nntime import time_this, timer_start, timer_end, export_timings


class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 7)

    # Don't tag boring functions
    def boring(self, x):
        return x + 1

    # tag the functions you want to time
    @time_this(
        name='stage1',  # override the auto generated names
        depth=2,        # assign a semantic call depth 
    )
    def forward(self, x):
        return self.conv(self.boring(x))
    
class Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 4)

    # tag the functions you want to time
    @time_this(
        name='stage2',  # override the auto generated names
        depth=2,        # assign a semantic call depth 
    )
    def forward(self, x):
        return self.linear(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1, self.stage2 = Stage1(), Stage2()

    # tag the functions you want to time
    @time_this(depth=1) # assign a semantic call depth 
    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.shape[0], -1)
        x = self.stage2(x)

        # also works with code segment along side with functions
        # need to give it a name in this case
        timer_start(self, 'softmax')
        x = F.softmax(x,  1)
        # assign a call depth at timer *end* for code snippets
        timer_end(self, 'softmax', depth=2)

        pred = x.max(0)[1]
        return pred

def main():
    dataset = [torch.randn([2, 3, 7, 7]) for _ in range(10)]
    model = Model()
    for data in dataset:
        _ = model(data)
    
    out_path = 'examples/call_depth.csv'
    # by default the timers will be sorted by call depths
    # but not shown in the export. Set show_depth=True to
    # see them in the export when you have marked them.
    export_timings(model, out_path, show_depth=True)
    print(f'Timing summary written to "{out_path}"')

if __name__ == '__main__':
    main()

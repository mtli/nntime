'''
This example shows how to disable torch.cuda.synchronize()
used for the timers. CUDA syncing is neccessary for accurate
timing of CUDA modules, but may introduce unwanted blocking
effect for the overall processing over a sequence of inputs.
You might also want to disable CUDA syncing for pure CPU
code.
'''

import torch
from torch import nn

from nntime import set_global_sync, time_this, export_timings
set_global_sync(False)
# import nntime to change the global settings before importing
# other modules that uses nntime


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 7)

    @time_this()
    def forward(self, x):
        return self.conv(x)

def main():
    dataset = [torch.randn([2, 3, 7, 7]) for _ in range(10)]
    model = Model()
    for data in dataset:
        _ = model(data)
    
    out_path = 'examples/sync_off.csv'
    export_timings(model, out_path)
    print(f'Timing summary written to "{out_path}"')

if __name__ == '__main__':
    main()

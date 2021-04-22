from os.path import isfile, dirname
from os import makedirs

import csv, functools
from time import perf_counter

import numpy as np
import torch


suffix = '_nntimes'
global_level = None
global_sync = False

def _update_counter(module, name, value, level):
    if hasattr(module, name):
        getattr(module, name)[1].append(value)
    else:
        setattr(module, name, (level, [value]))

def set_global_level(level):
    global global_level
    global_level = level

def get_global_level():
    return global_level

def set_global_sync(flag):
    global global_sync
    global_sync = flag

def get_global_sync():
    return global_sync

def time_this(name=None, level=None):
    """A decorator to store timing samples for a member function (e.g. forward in nn.Module)
    
    name: a name for the counter
    level: an integer that later can be used as a filter to select counters
    """
    def time_this_wrapper(old_func):
        if (global_level is not None) and (level is not None) and level < global_level:
            return old_func

        var_name = old_func.__name__ if name is None else name
        var_name += suffix

        # "wraps" is for keeping the name and the doc string of the decorated
        # function
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if global_sync:
                torch.cuda.synchronize()
            t_start = perf_counter()
            output = old_func(*args, **kwargs)
            if global_sync:
                torch.cuda.synchronize()
            t_end = perf_counter()

            _update_counter(args[0], var_name, t_end - t_start, level)
            return output
        return new_func
    return time_this_wrapper

def timer_start(module, name):
    """Mark the start of a piece of code in the member function for timing
    """
    if global_sync:
        torch.cuda.synchronize()
    setattr(module, name + '_start', perf_counter())

def timer_end(module, name, level=None):
    """Mark the end of a piece of code in the member function for timing
    """
    if global_sync:
        torch.cuda.synchronize()
    t_end = perf_counter()
    if (global_level is not None) and (level is not None) and level < global_level:
        return
    t_start = getattr(module, name + '_start')
    _update_counter(module, name + suffix, t_end - t_start, level)

def export_timings(
    model,
    out_path,
    overwrite=True,
    auto_mkdir=True,
    show_level=False,
    level=None,
    header=True,
    warmup=5,
    unit='ms', 
    fmt='.6g',
):
    """Export summary statistics for all timings in a PyTorch model (nn.Module) to a csv file
    """
    
    if not overwrite and isfile(out_path):
        return
        
    if unit == 'ms':
        cvt = lambda x: 1e3*x
    elif unit == 's':
        cvt = lambda x: x
    elif unit == 'μs' or unit == 'us':
        cvt = lambda x: 1e6*x
    elif unit == 'ns':
        cvt = lambda x: 1e9*x
    else:
        raise ValueError(f'Unknown time unit: {unit}')

    d = dirname(out_path)
    if d:
        makedirs(d, exist_ok=True)

    with open(out_path, 'w', newline='\n', encoding='utf-8') as f:
        w = csv.writer(f)
        if header:
            row = ['Level'] if show_level else []
            row += ['Item', f'Mean ({unit})', f'Std ({unit})',
                f'Min ({unit})', f'Max ({unit})']
            w.writerow(row)

        # multiple models
        if isinstance(model, list):
            models = model
        elif isinstance(model, tuple):
            assert len(model) == 2, 'If model is a tuple, it should be in the format of (<name>, <model>)'
            models = [model]
        else:
            models = [('', model)]
        for prefix, model in models:
            prefix = prefix + '.' if prefix else prefix
            for n, m in model.named_modules():
                for a in dir(m):
                    if a.endswith(suffix):
                        l, t = getattr(m, a)
                        if (level is not None) and (l is not None) and l < level:
                            continue
                        t = np.asarray(t)
                        if warmup:
                            t = t[warmup:]
                        name = prefix + (n + '.' if n else '')
                        name += a[:-len(suffix)]
                        assert len(t) > 0, f'Not enough samples for {name} (after discouting warmup)'

                        if show_level:
                            row = ['' if l is None else str(l)]
                        else:
                            row = []
                        row += [name,
                            f'{cvt(t.mean()):{fmt}}', f'{cvt(t.std(ddof=1)):{fmt}}',
                            f'{cvt(t.min()):{fmt}}', f'{cvt(t.max()):{fmt}}',
                        ]
                        w.writerow(row)

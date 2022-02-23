from os.path import isfile, dirname
from os import makedirs
from types import MethodType

import csv, functools
from time import perf_counter

import numpy as np
import torch


global_depth = None
global_sync = True
global_prefix = '_nntime_'

timer_list_name = global_prefix + 'timer_list'


def _update_counter(module, name, value, depth):
    if hasattr(module, timer_list_name):
        timer_list = getattr(module, timer_list_name)
        if name in timer_list:
            getattr(module, global_prefix + name)[1].append(value)
        else:
            timer_list.append(name)
            setattr(module, global_prefix + name, (depth, [value]))
    else:
        setattr(module, timer_list_name, [name])
        setattr(module, global_prefix + name, (depth, [value]))

def set_global_depth(depth):
    global global_depth
    global_depth = depth

def get_global_depth():
    return global_depth

def set_global_sync(flag):
    global global_sync
    global_sync = flag

def get_global_sync():
    return global_sync

def time_this(name=None, depth=None):
    """A decorator to store timing samples for a member function (e.g. forward in nn.Module)
    
    name: a name for the counter
    depth: an integer that later can be used as a filter to select counters
    """
    def time_this_wrapper(old_func):
        if (global_depth is not None) and (depth is not None) and depth > global_depth:
            return old_func

        var_name = old_func.__name__ if name is None else name

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

            _update_counter(args[0], var_name, t_end - t_start, depth)
            return output
        return new_func
    return time_this_wrapper

def timer_start(module, name):
    """Mark the start of a piece of code in the member function for timing
    """
    if global_sync:
        torch.cuda.synchronize()
    setattr(module, global_prefix + name + '_start', perf_counter())

def timer_end(module, name, depth=None):
    """Mark the end of a piece of code in the member function for timing
    """
    if global_sync:
        torch.cuda.synchronize()
    t_end = perf_counter()
    if (global_depth is not None) and (depth is not None) and depth > global_depth:
        return
    t_start = getattr(module, global_prefix + name + '_start')
    _update_counter(module, name, t_end - t_start, depth)

def time_tree(module, start_depth=0):
    """Add timers recursively to the forward function of the given module
       and all its submodules with automatic depth markers

       Note that each timer introduces a CUDA synchronization point when
       timing GPU code, and having too many such sync points may slow down
       pipelined execution, resulting in inaccurate measurement of the
       overall runtime.
    """
    # skip nn.ModuleList, which doesn't have a forward function
    if type(module) is not torch.nn.ModuleList:
        # Note1: this function is called after the module object is 
        # created, instead of at the time of creation, thus we need
        # MethodType here for binding
        
        # Note2: the module name will be generated during export, here
        # we append class types to disambiguate modules in
        # nn.ModuleList or nn.Sequential 

        module.forward = MethodType(
            time_this(f' - {module.__class__.__name__}', start_depth)
                (module.__class__.forward), module
        )
        
        start_depth += 1

    for m in module._modules.values():
        time_tree(m, start_depth)

def export_timings(
    model,
    out_path,
    overwrite=True,
    auto_mkdir=True,
    depth=None,
    sort_by_depth=False,
    show_depth=False,
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

    rows = []

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
        for m_name, module in model.named_modules():
            if hasattr(module, timer_list_name):
                for t_name in getattr(module, timer_list_name):
                    d, t = getattr(module, global_prefix + t_name)
                    if (depth is not None) and (d is not None) and d > depth:
                        continue
                    t = np.asarray(t)
                    if warmup:
                        t = t[warmup:]
                    name = prefix + m_name \
                         + ('.' if m_name and not t_name.startswith(' - ') else '') \
                         + t_name
                    # t_name.startswith(' - ') is for time_tree
                    assert len(t) > 0, f'Not enough samples for {name} (after discouting warmup)'

                    row = ['' if d is None else str(d), name,
                        f'{cvt(t.mean()):{fmt}}', f'{cvt(t.std(ddof=1)):{fmt}}',
                        f'{cvt(t.min()):{fmt}}', f'{cvt(t.max()):{fmt}}',
                    ]
                    rows.append(row)

    if sort_by_depth:
        rows = sorted(rows, key=lambda row: row[0])
    if not show_depth:
        rows = [row[1:] for row in rows]

    d = dirname(out_path)
    if d:
        makedirs(d, exist_ok=True)

    with open(out_path, 'w', newline='\n', encoding='utf-8') as f:
        w = csv.writer(f)
        if header:
            row = ['Depth'] if show_depth else []
            row += ['Item', f'Mean ({unit})', f'Std ({unit})',
                f'Min ({unit})', f'Max ({unit})']
            w.writerow(row)
        w.writerows(rows)

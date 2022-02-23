__version__ = '0.1.1'

from .timing import set_global_depth, get_global_depth, \
    set_global_sync, get_global_sync, \
    time_this, timer_start, timer_end, export_timings, \
    time_tree

__all__ = [
    'set_global_depth', 'get_global_depth',
    'set_global_sync', 'get_global_sync',
    'time_this', 'timer_start', 'timer_end',
    'export_timings', 'time_tree',
]

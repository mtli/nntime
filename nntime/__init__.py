__version__ = '0.0.1'

from .timing import set_global_level, get_global_level, \
    set_global_sync, get_global_sync, \
    time_this, timer_start, timer_end, export_timings

__all__ = [
    'set_global_level', 'get_global_level',
    'set_global_sync', 'get_global_sync',
    'time_this', 'timer_start', 'timer_end',
    'export_timings',
]


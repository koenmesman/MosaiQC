"""Optional pure-Python helpers around the C++ core."""

from . import _core

def sum_weights_safe(weights):
    return _core.sum_weights([float(x) for x in weights])

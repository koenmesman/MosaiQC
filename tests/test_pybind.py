import opt_core
from opt_core import _core

dev = _core.DeviceInfo()
dev.num_qubits = 5
dev.couplings = [(0, 1), (1, 2)]
print(_core.count_edges(dev))  # should be 2

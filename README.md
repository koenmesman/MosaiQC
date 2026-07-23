# MosaiQC

MosaiQC is a fast quantum circuit cutting package with a C++ optimization
backend exposed through Python as `MosaiQC`.

## Installation

Install the core package:

```bash
python -m pip install MosaiQC
```

Install optional Qiskit integration dependencies:

```bash
python -m pip install "MosaiQC[qiskit]"
```

## Usage

```python
import MosaiQC as mq

device = mq.DeviceInfo()
device.num_qubits = 5
device.couplings = [(0, 1), (1, 2)]

print(mq.count_edges(device))
```

Most APIs are exported from `MosaiQC` directly, including
`find_cutting_placement`, `refine_partition`, `CircuitGraph`, `TopologyGraph`,
and `DeviceInfo`. Advanced compiled bindings are available as `MosaiQC.native`.

## Build Locally

From this directory:

```bash
python -m pip install ".[dev]"
python -m build
python -m twine check dist/*
```

when available; the extension also builds without METIS support.

#from qiskit.providers.fake_provider import FakeBelemV2
from qiskit_ibm_runtime.fake_provider import FakeBelemV2

from qiskit_aer.noise import NoiseModel
import MosaiQC as mq

def test_backend_conversion_matches_noise_model():
    backend = FakeBelemV2()

    topo_from_backend = mq.TopologyGraph(backend=backend)

    model = NoiseModel.from_backend(backend)
    topo_from_model = mq.TopologyGraph(noise_model=model)

    assert set(topo_from_backend.qubits()) == set(topo_from_model.qubits())
    assert set(map(tuple, topo_from_backend.directed_edges())) == set(map(tuple, topo_from_model.directed_edges()))

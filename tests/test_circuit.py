from qiskit import QuantumCircuit
import MosaiQC as mq

if __name__ == "__main__":
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.h(0)          # ignored (1-qubit)
    qc.cx(0, 2)
    qc.cx(2, 0)

    qc.cx(0, 3)
    qc.x(2)          # ignored (1-qubit)
    qc.cx(1, 2)
    qc.cx(0, 1)

    print(qc)

    circuit = mq.CircuitGraph(qc)

    print("operations")
    print(circuit.ops())

    print("qubits")
    print(circuit.qubits())

    print("1q counts")
    print(circuit.oneq_counts())

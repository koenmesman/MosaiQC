import MosaiQC as mq

dev = mq.DeviceInfo()
dev.num_qubits = 5
dev.couplings = [(0, 1), (1, 2)]
print(mq.count_edges(dev))  # should be 2

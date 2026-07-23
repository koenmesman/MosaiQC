if __name__ == "__main__":
    import MosaiQC as mq

    try:
        from qiskit_ibm_runtime.fake_provider import (
            FakeBelemV2,
            FakeArmonkV2,
            FakeBogotaV2,
            FakeCairoV2,
            FakeJakartaV2,
        )
    except Exception as e:
        raise SystemExit(
            "qiskit_ibm_runtime fake providers are not available in this environment.\n"
            f"Import error: {e}"
        )

    backends = [FakeBelemV2(), FakeArmonkV2(), FakeBogotaV2(), FakeCairoV2(), FakeJakartaV2()]
    res = mq.assign_hardware(45, backends)
    print(res)

    res = mq.assign_hardware(50, backends)
    print(res)

    res = mq.assign_hardware(51, backends)
    print(res)
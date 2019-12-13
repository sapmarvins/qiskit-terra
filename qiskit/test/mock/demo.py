from qiskit import QuantumCircuit, transpile, schedule as build_schedule
from qiskit.test.mock import FakeNVCenters2Q

backend = FakeNVCenters2Q()
qc=QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)

Schedule = build_schedule(transpile(qc, backend), backend)
schedule.draw()
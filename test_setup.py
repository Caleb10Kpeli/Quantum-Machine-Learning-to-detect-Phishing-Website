import qiskit
import sklearn
import pandas
import numpy

print("✅ Qiskit version:", qiskit.__version__)
print("✅ Scikit-learn version:", sklearn.__version__)
print("✅ Pandas version:", pandas.__version__)
print("✅ NumPy version:", numpy.__version__)

# Test quantum simulator works
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sim = AerSimulator()
result = sim.run(qc).result()
counts = result.get_counts()
print("✅ Quantum simulation test passed! Result:", counts)
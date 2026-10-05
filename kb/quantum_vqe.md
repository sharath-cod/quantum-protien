# Quantum VQE in This App

## What VQE does
The Variational Quantum Eigensolver finds the lowest energy (ground state) of a Hamiltonian. A parameterized quantum circuit called the ansatz prepares a trial state, a quantum device or simulator measures the energy, and a classical optimizer updates the circuit parameters to push the energy down. It is a hybrid quantum-classical loop designed for noisy near-term hardware.

## The Hamiltonian
The Hamiltonian is the energy operator of the model. In this app it is built from the amino acid sequence as a sum of Pauli terms (ZZ interactions between qubits, plus single-qubit Z terms). Each qubit represents a segment of the chain. The terms encode hydrophobic interaction (favourable, lowers energy), electrostatic interaction (depends on charge), and hydrogen-bond-like contributions. The lowest eigenvalue is the model's minimum energy and its eigenvector is the predicted most stable configuration.

## Ansatz and optimizer
The ansatz is hardware-efficient: layers of RY rotations on every qubit followed by a linear chain of CNOT gates to create entanglement, repeated twice and finished with a last RY layer. The number of parameters is qubits times (reps + 1). The classical optimizer is COBYLA from scipy.

## Number of qubits
Qubit count grows with sequence length (about 1.5 times log2 of length+1) and is capped at 6 so the simulation finishes quickly on a free Render server. This is a coarse-grained model: each qubit covers a segment, not a single residue.

## Exact diagonalization and the convergence curve
For small qubit counts the app also computes the exact ground state with scipy.linalg.eigh. The VQE iterations are run to produce the convergence curve, which shows the energy dropping toward the exact minimum. If the VQE energy approaches the exact value, the ansatz is expressive enough for that Hamiltonian.

## Noise robustness
The noise endpoint re-runs the circuit with a depolarizing and thermal-relaxation noise model to mimic real hardware. If the energy stays close to the noiseless value the result is considered robust, and large drift means the answer would not survive on today's devices.

## Honest limitations
This is a simulation of a toy coarse-grained Hamiltonian on a classical computer. The energy unit shown (eV) is a model unit and is not a calibrated physical free energy. The result should not be read as a real folded structure or as proof of quantum advantage.

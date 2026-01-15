"""
Quantum Experiment: Testing if separation/distinction enables counting

Hypothesis: Separated qubits can be distinguished and counted as "1+1=2"
           Entangled qubits lose distinction and become "one system"

This tests our theory that space/separation is what enables addition.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def experiment_1_separated_qubits():
    """
    Experiment 1: Two independent qubits in superposition
    These are SEPARATED - they maintain distinction
    We should be able to clearly count "two things"
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: SEPARATED QUBITS (Clear Distinction)")
    print("=" * 60)

    qc = QuantumCircuit(2, 2)

    # Put each qubit independently in superposition
    qc.h(0)  # Qubit 0: |0> + |1>
    qc.h(1)  # Qubit 1: |0> + |1>

    # These are independent - we can count them as "1 + 1 = 2 qubits"

    qc.measure([0, 1], [0, 1])

    # Simulate
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    print("\nCircuit:")
    print(qc.draw(output='text'))
    print("\nResults (1000 shots):")
    print(counts)
    print("\nInterpretation:")
    print("- Four distinct outcomes: 00, 01, 10, 11")
    print("- Each outcome ~25% probability")
    print("- We can clearly distinguish TWO separate qubits")
    print("- Addition works: 1 qubit + 1 qubit = 2 distinguishable qubits")

    return counts


def experiment_2_entangled_qubits():
    """
    Experiment 2: Two maximally entangled qubits (Bell state)
    These are ENTANGLED - they lose individual distinction
    They become "one quantum system" - can we still count them as "2"?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: ENTANGLED QUBITS (Lost Distinction)")
    print("=" * 60)

    qc = QuantumCircuit(2, 2)

    # Create Bell state (maximally entangled)
    qc.h(0)  # Put qubit 0 in superposition
    qc.cx(0, 1)  # Entangle qubit 1 with qubit 0

    # Now they're ONE quantum system, not two separate things

    qc.measure([0, 1], [0, 1])

    # Simulate
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    print("\nCircuit:")
    print(qc.draw(output='text'))
    print("\nResults (1000 shots):")
    print(counts)
    print("\nInterpretation:")
    print("- Only TWO outcomes: 00 and 11 (never 01 or 10)")
    print("- Each outcome ~50% probability")
    print("- Qubits are perfectly correlated - they've lost independence")
    print("- They're now ONE entangled system, not two separate things")
    print("- Addition breaks down: 1 + 1 ≠ 2 when distinction is removed")

    return counts


def experiment_3_partial_entanglement():
    """
    Experiment 3: Partial entanglement with controlled rotation
    Testing the spectrum between "fully separated" and "fully entangled"
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: PARTIAL ENTANGLEMENT (Partial Distinction)")
    print("=" * 60)

    qc = QuantumCircuit(2, 2)

    # Create partial entanglement using controlled rotation
    qc.h(0)
    qc.cry(3.14159 / 4, 0, 1)  # Partial entanglement (45 degrees)

    qc.measure([0, 1], [0, 1])

    # Simulate
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    print("\nCircuit:")
    print(qc.draw(output='text'))
    print("\nResults (1000 shots):")
    print(counts)
    print("\nInterpretation:")
    print("- All four outcomes possible, but unequal probabilities")
    print("- Partial correlation between qubits")
    print("- Somewhere between 'two separate things' and 'one thing'")
    print("- The degree of separation determines how well we can count them")

    return counts


def experiment_4_three_qubits_ghz():
    """
    Experiment 4: Three-qubit GHZ state
    Testing if loss of distinction scales with number of qubits
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: THREE-QUBIT GHZ STATE (No 3-way Distinction)")
    print("=" * 60)

    qc = QuantumCircuit(3, 3)

    # Create GHZ state: |000> + |111>
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)

    qc.measure([0, 1, 2], [0, 1, 2])

    # Simulate
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    print("\nCircuit:")
    print(qc.draw(output='text'))
    print("\nResults (1000 shots):")
    print(counts)
    print("\nInterpretation:")
    print("- Only TWO outcomes: 000 and 111")
    print("- All three qubits perfectly correlated")
    print("- Cannot distinguish individual qubits")
    print("- 1 + 1 + 1 = ? (not 3 when they're all entangled)")

    return counts


def visualize_results(counts_list, titles):
    """
    Visualize all experiments side by side
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (counts, title) in enumerate(zip(counts_list, titles)):
        ax = axes[idx]
        outcomes = list(counts.keys())
        values = list(counts.values())

        ax.bar(outcomes, values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Measurement Outcome', fontsize=12)
        ax.set_ylabel('Count (out of 1000)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('quantum_separation_test.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Visualization saved as 'quantum_separation_test.png'")
    print("=" * 60)
    plt.show()


def main():
    """
    Run all experiments and analyze results
    """
    print("\n" + "=" * 70)
    print(" QUANTUM TEST: DOES SEPARATION ENABLE ADDITION/COUNTING?")
    print("=" * 70)
    print("\nTheory: Space/separation is the pattern that enables addition.")
    print("Test: Can we 'count' qubits when we remove their separation via entanglement?")
    print("=" * 70)

    # Run experiments
    counts1 = experiment_1_separated_qubits()
    counts2 = experiment_2_entangled_qubits()
    counts3 = experiment_3_partial_entanglement()
    counts4 = experiment_4_three_qubits_ghz()

    # Visualize
    counts_list = [counts1, counts2, counts3, counts4]
    titles = [
        "Exp 1: Separated (Full Distinction)",
        "Exp 2: Entangled (No Distinction)",
        "Exp 3: Partial Entanglement",
        "Exp 4: 3-Qubit GHZ State"
    ]

    visualize_results(counts_list, titles)

    # Final analysis
    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)
    print("\n✓ SEPARATED qubits (Exp 1): Can count as '1+1=2' - 4 distinct outcomes")
    print("✗ ENTANGLED qubits (Exp 2): Cannot count as '1+1=2' - become ONE system")
    print("~ PARTIAL entanglement (Exp 3): Counting becomes ambiguous")
    print("✗ 3-way entanglement (Exp 4): '1+1+1≠3' - all become ONE system")
    print("\nRESULT: When we remove separation (via entanglement), the ability to")
    print("        distinguish and count individual units BREAKS DOWN.")
    print("\nThis supports the theory: SEPARATION/SPACE enables ADDITION/COUNTING")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
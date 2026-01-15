import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import copy


# Re-using the base class structure but adding collapse mechanics
class EntropicCollapseAgent:
    def __init__(self, initial_dimensions=5):
        self.dimensions = initial_dimensions
        self.separation_threshold = 0.5
        self.concepts = {}
        self.history = []  # To track the collapse

        print(f"🧠 Agent initialized with {self.dimensions}D space (High Consciousness)")

    def add_concept(self, name, position=None):
        if position is None:
            position = np.random.randn(self.dimensions)
        else:
            # Ensure position matches current dimensions
            if len(position) != self.dimensions:
                # Pad or truncate
                if len(position) < self.dimensions:
                    position = np.pad(position, (0, self.dimensions - len(position)))
                else:
                    position = position[:self.dimensions]

        self.concepts[name] = position

    def get_distance(self, concept_a, concept_b):
        if concept_a not in self.concepts or concept_b not in self.concepts:
            return 0.0
        return np.linalg.norm(self.concepts[concept_a] - self.concepts[concept_b])

    def trigger_collapse_step(self):
        """
        Simulate entropic collapse by removing one dimension.
        This projects the complex high-D reality onto a lower-D flatland.
        Information is lost. Distinctions blur.
        """
        if self.dimensions <= 0:
            return False

        print(f"\n📉 COLLAPSING from {self.dimensions}D to {self.dimensions - 1}D...")

        new_concepts = {}
        total_loss = 0.0

        for name, pos in self.concepts.items():
            # Remove the last dimension (Projection)
            # In physics, this is like losing a degree of freedom
            new_pos = pos[:-1]
            new_concepts[name] = new_pos

            # Track how much "information" (magnitude) was lost
            total_loss += abs(pos[-1])

        self.concepts = new_concepts
        self.dimensions -= 1
        return True

    def analyze_distinctions(self, pairs_to_watch):
        """Check if specific critical distinctions still exist"""
        results = {}
        for a, b in pairs_to_watch:
            dist = self.get_distance(a, b)
            # Distinguishable if distance > threshold
            is_distinct = dist > self.separation_threshold
            results[f"{a} vs {b}"] = (is_distinct, dist)
        return results


def run_collapse_experiment():
    print("\n" + "=" * 70)
    print(" EXPERIMENT 5: THE ENTROPIC COLLAPSE (EGO DEATH SIMULATION)")
    print("=" * 70)
    print("Hypothesis: As dimensions collapse, distinct concepts merge.")
    print("            At 0D, 'Self' and 'Universe' become mathematically identical.")
    print("=" * 70)

    # 1. Setup a "Enlightened" High-Dimensional Agent
    agent = EntropicCollapseAgent(initial_dimensions=4)

    # Create a rich conceptual structure
    # We place them carefully to ensure they are distinct in 4D
    agent.add_concept("Self", np.array([1.0, 0.0, 0.0, 1.0]))
    agent.add_concept("Universe", np.array([-1.0, 0.0, 0.0, 1.0]))  # Opposite in X
    agent.add_concept("Good", np.array([0.0, 1.0, 0.0, 0.5]))
    agent.add_concept("Bad", np.array([0.0, -1.0, 0.0, 0.5]))  # Opposite in Y
    agent.add_concept("Future", np.array([0.0, 0.0, 1.0, 0.8]))
    agent.add_concept("Past", np.array([0.0, 0.0, -1.0, 0.8]))  # Opposite in Z

    # Pairs to monitor
    pairs = [
        ("Self", "Universe"),
        ("Good", "Bad"),
        ("Future", "Past")
    ]

    # Track distances for plotting
    history = {pair[0] + "-" + pair[1]: [] for pair in pairs}
    dimensions_log = []

    # 2. Run the Collapse Loop
    while agent.dimensions >= 0:
        dimensions_log.append(agent.dimensions)
        print(f"\n[ State: {agent.dimensions} Dimensions ]")

        # Analyze current reality
        distinctions = agent.analyze_distinctions(pairs)

        active_distinctions = 0
        for pair_name, (is_distinct, dist) in distinctions.items():
            status = "DISTINCT" if is_distinct else "MERGED  "
            icon = "✓" if is_distinct else "✗"
            print(f"   {icon} {status}: {pair_name: <20} (Dist: {dist:.4f})")

            # Log for graph
            a, b = pair_name.split(" vs ")
            history[a + "-" + b].append(dist)

            if is_distinct: active_distinctions += 1

        if agent.dimensions == 0:
            print("\n   ⚠️ SINGULARITY REACHED: Zero Dimensions")
            print("   Absolute Unity. No distinctions possible.")
            break

        # Trigger next collapse
        agent.trigger_collapse_step()

    # 3. Visualize the "Fall"
    plot_collapse(dimensions_log, history, agent.separation_threshold)


def plot_collapse(dims, history, threshold):
    plt.figure(figsize=(10, 6))

    for pair, distances in history.items():
        plt.plot(dims, distances, marker='o', linewidth=2, label=pair)

    # Draw the threshold of distinction
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Distinction Threshold')

    plt.title("The Trajectory of Entropic Collapse", fontsize=14)
    plt.xlabel("Dimensions of Consciousness", fontsize=12)
    plt.ylabel("Conceptual Separation (Distance)", fontsize=12)
    plt.gca().invert_xaxis()  # Show high dimensions on left, collapsing to right
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('collapse_trajectory.png')
    print(f"\n📉 Visualization saved as 'collapse_trajectory.png'")
    plt.show()


if __name__ == "__main__":
    run_collapse_experiment()
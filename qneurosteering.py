import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


class NeuroSteeringAgent:
    def __init__(self):
        self.dimensions = 3
        self.concepts = {}
        self.trajectory = []

        # The "Depression Bias": An internal weight pulling thoughts "down"
        # In this simulation, Y-axis represents Valence (Emotion)
        # Negative Y = Sad, Positive Y = Happy
        self.internal_bias = np.array([0.0, -0.8, 0.0])

        # Word associations to seed the geometry
        self.lexicon = {
            'future': np.array([0.5, 0.1, 0.8]),
            'work': np.array([0.2, 0.0, 0.1]),
            'life': np.array([0.0, 0.1, 0.0]),
            'myself': np.array([0.0, 0.0, 0.0]),
            'world': np.array([0.5, 0.0, 0.0]),
            'relationship': np.array([0.3, 0.2, 0.0]),
            'memory': np.array([-0.2, 0.0, -0.5])
        }

    def reset(self):
        self.concepts = {}
        self.trajectory = []

    def process_stream(self, thought_stream, steering_vector=None):
        """
        Process thoughts.
        Normal mode: Apply internal bias (Depression).
        Steered mode: Apply Steering Vector (Therapy/Control).
        """
        for word in thought_stream:
            if word in self.lexicon:
                # 1. Get base meaning (Platonic ideal of the concept)
                base_vec = self.lexicon[word]

                # 2. Apply Internal State (The "Filter")
                # The agent distorts the concept based on its mood
                perceived_vec = base_vec + self.internal_bias + (np.random.randn(3) * 0.1)

                # 3. Apply Neuro-Steering (if active)
                # RepE: "Adding the vector to the representation"
                if steering_vector is not None:
                    perceived_vec += steering_vector

                self.concepts[word] = perceived_vec
                self.trajectory.append(perceived_vec)

    def calculate_emotional_state(self):
        """
        Measure the average position on the Y-Axis (Valence).
        """
        if not self.concepts: return 0.0

        # Average Y-coordinate of all concepts
        vectors = np.array(list(self.concepts.values()))
        avg_valence = np.mean(vectors[:, 1])  # Index 1 is Y-axis
        return avg_valence


def run_neuro_steering():
    print("\n" + "=" * 70)
    print(" APPLICATION 3: NEURO-STEERING (RepE SIMULATION)")
    print("=" * 70)
    print("Goal: Mathematically 'cure' a depressed agent by injecting a specific")
    print("      geometric vector, proving Emotion is a Direction.")
    print("=" * 70)

    # The Input Stream (Ambiguous concepts that can be interpreted good or bad)
    stream = ['myself', 'life', 'work', 'future', 'relationship', 'world', 'memory']

    agent = NeuroSteeringAgent()

    # --- PHASE 1: THE DEPRESSED STATE (Baseline) ---
    print("\n1. RUNNING BASELINE (Depressed State)...")
    agent.process_stream(stream, steering_vector=None)

    mood_base = agent.calculate_emotional_state()
    print(f"   📉 Average Valence (Mood): {mood_base:.2f}")
    print("   Interpretation: Concepts are clustered in the negative Y-plane.")

    # Capture state for plotting
    concepts_base = agent.concepts.copy()

    # --- PHASE 2: CALCULATING THE CURE ---
    # In RepE, we find the "direction" of the desired behavior.
    # Here, we calculate the vector needed to push the average Y to +0.5
    target_valence = 0.8
    correction_needed = target_valence - mood_base

    # The Steering Vector: Zero on X/Z, strong push on Y
    steering_vec = np.array([0.0, correction_needed, 0.0])

    print(f"\n2. EXTRACTING STEERING VECTOR...")
    print(f"   💉 Calculated Injection: {steering_vec}")
    print("   (This vector represents 'Hope/Positivity' in this vector space)")

    # --- PHASE 3: APPLYING STEERING (The Intervention) ---
    print("\n3. APPLYING NEURO-STEERING...")
    agent.reset()
    agent.process_stream(stream, steering_vector=steering_vec)

    mood_steered = agent.calculate_emotional_state()
    print(f"   📈 New Valence (Mood): {mood_steered:.2f}")
    print("   Interpretation: The agent's reality has been geometrically shifted.")

    # Capture state for plotting
    concepts_steered = agent.concepts.copy()

    # --- VISUALIZATION ---
    visualize_steering(concepts_base, concepts_steered, stream)


def visualize_steering(base, steered, labels):
    fig = plt.figure(figsize=(12, 6))

    # Plot 1: Depressed Reality
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mind(ax1, base, "Subject A: Depressed Geometry", 'red')

    # Plot 2: Steered Reality
    ax2 = fig.add_subplot(122, projection='3d')
    plot_mind(ax2, steered, "Subject A + Steering Vector", 'green')

    plt.tight_layout()
    plt.savefig('neuro_steering_result.png')
    print("\n📊 Visualization saved as 'neuro_steering_result.png'")
    plt.show()


def plot_mind(ax, concepts, title, color):
    # Extract coords
    vecs = np.array(list(concepts.values()))
    xs, ys, zs = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    # Draw the "Zero Plane" (Neutral Emotion)
    xx, zz = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    yy = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

    # Scatter points
    ax.scatter(xs, ys, zs, c=color, s=100, alpha=0.8)

    # Connect to "Self" (0,0,0) for context
    for i, word in enumerate(concepts):
        x, y, z = concepts[word]
        ax.text(x, y, z + 0.1, word, fontsize=9)
        # Drop lines to the emotional plane (Y=0) to show "depth" of emotion
        ax.plot([x, x], [y, 0], [z, z], 'k--', alpha=0.3)

    ax.set_xlabel('Self <-> World')
    ax.set_ylabel('Sad <-> Happy')
    ax.set_zlabel('Past <-> Future')
    ax.set_title(title)

    # Fix view to emphasize the Y-axis shift
    ax.view_init(elev=20, azim=-60)
    ax.set_ylim(-1.5, 1.5)


if __name__ == "__main__":
    run_neuro_steering()
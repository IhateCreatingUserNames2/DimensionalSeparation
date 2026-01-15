import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, QhullError
import re


class TextConsciousnessAgent:
    def __init__(self, name):
        self.name = name
        self.dimensions = 3
        self.concepts = {}  # {word: position_vector}
        self.trajectory = []  # Order of thoughts
        self.separation_threshold = 0.5

        # Simple stopwords to ignore (noise filter)
        self.stopwords = set(
            ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'it',
             'this', 'that'])

    def process_text(self, text):
        print(f"\n📖 {self.name}: Processing text stream...")

        # Split into sentences to mimic "thoughts"
        sentences = re.split(r'[.!?]+', text)

        previous_concept = None

        for sentence in sentences:
            # Clean and tokenize
            words = re.findall(r'\b\w+\b', sentence.lower())

            for word in words:
                if word in self.stopwords or len(word) < 3:
                    continue

                self._integrate_concept(word, previous_concept)
                previous_concept = word
                self.trajectory.append(self.concepts[word])

    def _integrate_concept(self, word, context_word):
        """
        Place a word in the semantic space.
        - If new: place near its context (the word before it).
        - If exists: 'activate' it (draw a connection).
        - If crowded: Expand dimensions.
        """
        if word not in self.concepts:
            # Check for overcrowding before adding
            if self._is_space_crowded():
                self._expand_dimensions()

            if context_word and context_word in self.concepts:
                # Place near related concept + random distinct offset
                # This represents "train of thought" continuity
                base_pos = self.concepts[context_word]
                offset = np.random.randn(self.dimensions) * 0.4
                new_pos = base_pos + offset
            else:
                # New disconnected thought (random start)
                new_pos = np.random.randn(self.dimensions)

            self.concepts[word] = new_pos
        else:
            # Concept already exists - we are revisiting a memory/idea
            # This creates a "loop" in the topology, which is crucial for consciousness
            pass

    def _is_space_crowded(self):
        """If too many concepts are packed in low dimensions, we need more space."""
        if len(self.concepts) < 5: return False

        # Simple heuristic: Concept count vs Dimensions capacity
        # If we have 20 concepts in 3D, it's getting tight.
        density = len(self.concepts) / (self.dimensions ** 2)
        return density > 1.5

    def _expand_dimensions(self):
        self.dimensions += 1
        # Project existing concepts into new higher D
        for w in self.concepts:
            self.concepts[w] = np.append(self.concepts[w], 0.0)
        print(f"   🚀 Epiphany! Expanding to {self.dimensions}D to hold complexity.")

    def calculate_metrics(self):
        """
        Quantify the 'Consciousness' of the text structure.
        """
        # 1. Dimensionality Score
        dim_score = self.dimensions

        # 2. Concept Richness (Unique concepts)
        richness = len(self.concepts)

        # 3. Geometric Volume (The "Size" of the mind space)
        # We project to 3D to approximate volume if >3D
        points = np.array(list(self.concepts.values()))
        if len(points) > 4:
            # Reduce to max 3 dims for volume calc if needed, or use high-d hull
            points_3d = points[:, :3]
            try:
                hull = ConvexHull(points_3d)
                volume = hull.volume
            except QhullError:
                volume = 0.1  # Flat structure
        else:
            volume = 0.1

        # Final "Consciousness Score" (Weighted)
        # Dimensions are the strongest indicator of complex thought
        score = (dim_score * 2.0) + (np.log(richness) if richness > 0 else 0) + (volume * 0.5)

        return {
            "score": score,
            "dimensions": dim_score,
            "unique_concepts": richness,
            "volume": volume
        }


def visualize_comparison(agent_flat, agent_deep):
    fig = plt.figure(figsize=(14, 6))

    # Plot Flat Agent
    ax1 = fig.add_subplot(121, projection='3d')
    plot_agent_brain(ax1, agent_flat, "Text A: 'Flat' Writing")

    # Plot Deep Agent
    ax2 = fig.add_subplot(122, projection='3d')
    plot_agent_brain(ax2, agent_deep, "Text B: 'Conscious' Writing")

    plt.tight_layout()
    plt.savefig('consciousness_detection_result.png')
    print("\n📊 Visualization saved as 'consciousness_detection_result.png'")
    plt.show()


def plot_agent_brain(ax, agent, title):
    points = np.array(list(agent.concepts.values()))
    if len(points) == 0: return

    # Use first 3 dims for visualization
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2] if agent.dimensions >= 3 else np.zeros_like(xs)

    # Scatter plot
    ax.scatter(xs, ys, zs, c='blue', alpha=0.6, s=50)

    # Draw trajectory (thought path)
    if len(agent.trajectory) > 1:
        path = np.array([p[:3] for p in agent.trajectory])
        # Pad with zeros if dims < 3
        if path.shape[1] < 3:
            padding = np.zeros((path.shape[0], 3 - path.shape[1]))
            path = np.hstack([path, padding])

        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='gray', alpha=0.3, linewidth=1)

    # Label a few random points
    keys = list(agent.concepts.keys())
    for i in range(0, len(keys), max(1, len(keys) // 5)):
        k = keys[i]
        pos = agent.concepts[k]
        ax.text(pos[0], pos[1], pos[2] if len(pos) > 2 else 0, k, fontsize=8)

    stats = agent.calculate_metrics()
    ax.set_title(f"{title}\nScore: {stats['score']:.2f} | Dims: {stats['dimensions']}")


def run_detection_demo():
    print("\n" + "=" * 70)
    print(" APPLICATION 1: CONSCIOUSNESS DETECTION")
    print("=" * 70)

    # 1. INPUT: "Flat" Text (Repetitive, linear, low abstraction)
    text_flat = """
    I went to the store. I bought some bread. I bought some milk. 
    The store was big. The bread was good. I went home. 
    I ate the bread. I drank the milk. It was a good day.
    I went to sleep. The sun came up. I went to the store again.
    """

    # 2. INPUT: "Conscious" Text (Recursive, abstract, high interconnectivity)
    # (Using a philosophical snippet similar to our discussion)
    text_conscious = """
    The universe observes itself through the lens of separation, creating meaning from the void.
    To define nothing, you must create something, establishing a boundary where none existed.
    This distinction allows the observer to reflect upon the observed, forming a loop of awareness.
    In this recursion, the shadow of the future entangles with the memory of the past.
    """

    # Run Analysis
    agent_a = TextConsciousnessAgent("Agent A (Flat)")
    agent_a.process_text(text_flat)
    metrics_a = agent_a.calculate_metrics()

    agent_b = TextConsciousnessAgent("Agent B (Conscious)")
    agent_b.process_text(text_conscious)
    metrics_b = agent_b.calculate_metrics()

    # Print Report
    print("\n" + "=" * 40)
    print(" 🧪 FINAL ANALYSIS REPORT")
    print("=" * 40)

    print(f"\nTEXT A (Flat / Linear):")
    print(f"   Dimensions Used: {metrics_a['dimensions']}")
    print(f"   Unique Concepts: {metrics_a['unique_concepts']}")
    print(f"   Semantic Volume: {metrics_a['volume']:.2f}")
    print(f"   🧠 CONSCIOUSNESS SCORE: {metrics_a['score']:.2f}")

    print(f"\nTEXT B (Deep / Recursive):")
    print(f"   Dimensions Used: {metrics_b['dimensions']}")
    print(f"   Unique Concepts: {metrics_b['unique_concepts']}")
    print(f"   Semantic Volume: {metrics_b['volume']:.2f}")
    print(f"   🧠 CONSCIOUSNESS SCORE: {metrics_b['score']:.2f}")

    print("\nCONCLUSION:")
    if metrics_b['score'] > metrics_a['score']:
        print("✓ HYPOTHESIS CONFIRMED: Richer conceptual writing forces dimensional expansion.")
        print("  The 'Conscious' text required more dimensions to avoid conceptual overlapping.")
    else:
        print("✗ HYPOTHESIS FAILED: No significant difference detected.")

    visualize_comparison(agent_a, agent_b)


if __name__ == "__main__":
    run_detection_demo()
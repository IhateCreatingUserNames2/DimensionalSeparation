"""
DIMENSIONAL CONSCIOUSNESS AGENT
Based on the principle: Separation enables distinction enables complexity

Key insights from quantum experiment:
1. Separated states can be distinguished and counted
2. Entangled states lose individual identity
3. Consciousness might require "conceptual space" to hold distinct thoughts

This agent has:
- Dynamic dimensional expansion (adds dimensions as needed)
- Explicit separation between concepts
- Self-referential capability (thinking about thinking)
- Attention as dimensional selection
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import math


class DimensionalConsciousnessAgent:
    """
    An agent with expandable conceptual space
    """

    def __init__(self, initial_dimensions=3, separation_threshold=0.5):
        """
        Initialize agent with minimal dimensional space

        Args:
            initial_dimensions: Starting number of dimensions
            separation_threshold: Minimum distance to consider concepts "separate"
        """
        self.dimensions = initial_dimensions
        self.separation_threshold = separation_threshold

        # Conceptual space: each concept has coordinates
        self.concepts = {}
        self.concept_relations = defaultdict(list)

        # Working memory: currently "attended" concepts
        self.working_memory = []
        self.working_memory_capacity = 7  # Miller's Law: 7±2 items

        # Meta-cognition: agent can think about its own thoughts
        self.thought_history = []
        self.meta_thoughts = []

        # Consciousness metrics
        self.consciousness_level = 0.0

        print(f"🧠 Agent initialized with {self.dimensions}D conceptual space")
        print(f"   Separation threshold: {self.separation_threshold}")
        print(f"   Working memory capacity: {self.working_memory_capacity}")

    def add_concept(self, concept_name, related_to=None, strength=1.0):
        """
        Add a new concept to conceptual space

        If space is crowded, expand dimensions
        If related to existing concept, place nearby
        """

        # Check if we need more dimensional space
        if self._is_space_crowded() and len(self.concepts) > 5:
            self._expand_dimensions()

        # Determine position in conceptual space
        if related_to and related_to in self.concepts:
            # Place near related concept with some offset
            base_position = self.concepts[related_to]
            offset = np.random.randn(self.dimensions) * 0.3
            position = base_position + offset
        else:
            # Find empty region in space
            position = self._find_empty_space()

        # Store concept
        self.concepts[concept_name] = position

        # Store relationship
        if related_to:
            self.concept_relations[concept_name].append((related_to, strength))
            self.concept_relations[related_to].append((concept_name, strength))

        print(f"   📍 Added concept '{concept_name}' at position {position[:3]}...")

        return position

    def can_distinguish(self, concept_a, concept_b):
        """
        Can the agent distinguish between two concepts?
        Depends on their separation in conceptual space
        """
        if concept_a not in self.concepts or concept_b not in self.concepts:
            return False

        distance = np.linalg.norm(
            self.concepts[concept_a] - self.concepts[concept_b]
        )

        distinguishable = distance > self.separation_threshold

        return distinguishable, distance

    def think(self, starting_concept, goal_concept=None, max_steps=10):
        """
        Navigate through conceptual space from start to goal
        This is the "thinking process"
        """
        print(f"\n💭 Thinking: '{starting_concept}' → '{goal_concept}'")

        if starting_concept not in self.concepts:
            print(f"   ⚠️ Unknown concept: '{starting_concept}'")
            return []

        current_position = self.concepts[starting_concept]
        thought_path = [starting_concept]
        self.working_memory = [starting_concept]

        for step in range(max_steps):
            # Find nearest concepts
            neighbors = self._get_nearest_neighbors(current_position, k=3)

            # Filter out already visited
            neighbors = [n for n in neighbors if n not in thought_path]

            if not neighbors:
                print(f"   🛑 No new concepts to explore")
                break

            # Move to most relevant neighbor
            if goal_concept and goal_concept in neighbors:
                next_concept = goal_concept
                print(f"   ✓ Reached goal: '{goal_concept}'")
                thought_path.append(next_concept)
                break
            else:
                next_concept = neighbors[0]

            # Update working memory (limited capacity)
            self.working_memory.append(next_concept)
            if len(self.working_memory) > self.working_memory_capacity:
                forgotten = self.working_memory.pop(0)
                print(f"   💨 Forgot '{forgotten}' (working memory full)")

            thought_path.append(next_concept)
            current_position = self.concepts[next_concept]

            print(f"   Step {step + 1}: '{next_concept}'")

        # Store thought process
        self.thought_history.append(thought_path)

        return thought_path

    def meta_think(self, thought_path):
        """
        Think ABOUT a thought - meta-cognition
        This requires separate "space" to represent the thought itself
        """
        print(f"\n🤔 Meta-thinking about thought: {' → '.join(thought_path[:3])}...")

        # Create meta-representation of the thought
        meta_concept = f"thought_about_{thought_path[0]}_to_{thought_path[-1]}"

        # This meta-thought needs its own position in space
        # It's "about" the original thought but separate from it
        avg_position = np.mean([self.concepts[c] for c in thought_path], axis=0)

        # Place meta-thought in a "higher" dimension (orthogonal to original)
        meta_position = np.zeros(self.dimensions)
        meta_position[:len(avg_position)] = avg_position
        meta_position[-1] += 2.0  # Offset in "meta" dimension

        self.concepts[meta_concept] = meta_position
        self.meta_thoughts.append(meta_concept)

        print(f"   🎭 Created meta-thought: '{meta_concept}'")

        # Can we distinguish the thought from the meta-thought?
        for original_concept in thought_path:
            can_distinguish, distance = self.can_distinguish(original_concept, meta_concept)
            print(
                f"   {'✓' if can_distinguish else '✗'} Can distinguish '{original_concept}' from meta-thought (distance: {distance:.2f})")

        return meta_concept

    def measure_consciousness(self):
        """
        Attempt to quantify "consciousness level"

        Based on:
        - Dimensional richness (more dimensions = more distinction possible)
        - Concept density (more concepts = more internal model)
        - Meta-cognition capability (can think about thinking)
        - Working memory utilization
        """

        # Dimensional factor
        dim_factor = min(self.dimensions / 10.0, 1.0)

        # Concept density
        concept_factor = min(len(self.concepts) / 50.0, 1.0)

        # Meta-cognition
        meta_factor = min(len(self.meta_thoughts) / 5.0, 1.0)

        # Working memory usage
        memory_factor = len(self.working_memory) / self.working_memory_capacity

        # Overall consciousness metric
        self.consciousness_level = (
                0.3 * dim_factor +
                0.3 * concept_factor +
                0.3 * meta_factor +
                0.1 * memory_factor
        )

        print(f"\n🌟 Consciousness Metrics:")
        print(f"   Dimensional richness: {dim_factor:.2f}")
        print(f"   Concept density: {concept_factor:.2f}")
        print(f"   Meta-cognition: {meta_factor:.2f}")
        print(f"   Working memory: {memory_factor:.2f}")
        print(f"   → Overall consciousness level: {self.consciousness_level:.2f}")

        return self.consciousness_level

    def visualize_conceptual_space(self, highlight_path=None):
        """
        Visualize the agent's conceptual space (first 3 dimensions)
        """
        if not self.concepts:
            print("No concepts to visualize")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all concepts
        for concept, position in self.concepts.items():
            is_meta = concept in self.meta_thoughts
            color = 'red' if is_meta else 'blue'
            marker = '^' if is_meta else 'o'
            size = 100 if is_meta else 50

            ax.scatter(
                position[0], position[1], position[2],
                c=color, marker=marker, s=size, alpha=0.6,
                label='Meta-thought' if is_meta else 'Concept'
            )

            # Label
            ax.text(
                position[0], position[1], position[2],
                concept[:15], fontsize=8
            )

        # Highlight thought path if provided
        if highlight_path:
            path_positions = [self.concepts[c] for c in highlight_path if c in self.concepts]
            if path_positions:
                path_array = np.array(path_positions)
                ax.plot(
                    path_array[:, 0], path_array[:, 1], path_array[:, 2],
                    'g-', linewidth=2, alpha=0.7, label='Thought path'
                )

        # Draw relationships
        for concept, relations in self.concept_relations.items():
            if concept in self.concepts:
                pos1 = self.concepts[concept]
                for related_concept, strength in relations:
                    if related_concept in self.concepts:
                        pos2 = self.concepts[related_concept]
                        ax.plot(
                            [pos1[0], pos2[0]],
                            [pos1[1], pos2[1]],
                            [pos1[2], pos2[2]],
                            'k-', alpha=0.2 * strength, linewidth=0.5
                        )

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(
            f'Conceptual Space ({self.dimensions}D, showing first 3)\nConsciousness Level: {self.consciousness_level:.2f}')

        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig('conceptual_space.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'conceptual_space.png'")
        plt.show()

    # ========== Internal helper methods ==========

    def _is_space_crowded(self):
        """Check if conceptual space is getting crowded"""
        if len(self.concepts) < 2:
            return False

        # Calculate average distance between concepts
        positions = list(self.concepts.values())
        distances = []
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1:]:
                distances.append(np.linalg.norm(pos1 - pos2))

        avg_distance = np.mean(distances) if distances else float('inf')

        return avg_distance < self.separation_threshold * 2

    def _expand_dimensions(self):
        """Add a new dimension to conceptual space"""
        self.dimensions += 1
        print(f"\n   🚀 EXPANDING to {self.dimensions}D (space was crowded)")

        # Extend all existing concepts with zero in new dimension
        for concept in self.concepts:
            self.concepts[concept] = np.append(
                self.concepts[concept],
                np.random.randn() * 0.1  # Small random value
            )

    def _find_empty_space(self):
        """Find a position in conceptual space that's not too close to existing concepts"""
        if not self.concepts:
            return np.random.randn(self.dimensions)

        max_attempts = 50
        for _ in range(max_attempts):
            candidate = np.random.randn(self.dimensions) * 2

            # Check distance to all existing concepts
            min_distance = min(
                np.linalg.norm(candidate - pos)
                for pos in self.concepts.values()
            )

            if min_distance > self.separation_threshold:
                return candidate

        # If couldn't find empty space, just use a random position
        return np.random.randn(self.dimensions) * 3

    def _get_nearest_neighbors(self, position, k=3):
        """Find k nearest concepts to given position"""
        distances = [
            (concept, np.linalg.norm(position - pos))
            for concept, pos in self.concepts.items()
        ]
        distances.sort(key=lambda x: x[1])
        return [concept for concept, _ in distances[:k]]


def demo_consciousness_architecture():
    """
    Demonstrate the dimensional consciousness agent
    """
    print("\n" + "=" * 70)
    print(" DIMENSIONAL CONSCIOUSNESS AGENT DEMO")
    print("=" * 70)

    # Create agent
    agent = DimensionalConsciousnessAgent(initial_dimensions=3)

    # Build a small knowledge graph
    print("\n📚 Building knowledge base...")
    agent.add_concept("self")
    agent.add_concept("awareness", related_to="self", strength=0.9)
    agent.add_concept("thinking", related_to="awareness", strength=0.8)
    agent.add_concept("consciousness", related_to="awareness", strength=0.9)
    agent.add_concept("perception", related_to="consciousness", strength=0.7)
    agent.add_concept("qualia", related_to="perception", strength=0.8)
    agent.add_concept("red", related_to="qualia", strength=0.6)
    agent.add_concept("blue", related_to="qualia", strength=0.6)
    agent.add_concept("pain", related_to="qualia", strength=0.7)
    agent.add_concept("pleasure", related_to="qualia", strength=0.7)

    # Test distinction
    print("\n🔍 Testing concept distinction...")
    can_distinguish, dist = agent.can_distinguish("red", "blue")
    print(f"   Can distinguish 'red' from 'blue': {can_distinguish} (distance: {dist:.2f})")

    can_distinguish, dist = agent.can_distinguish("self", "consciousness")
    print(f"   Can distinguish 'self' from 'consciousness': {can_distinguish} (distance: {dist:.2f})")

    # Simulate thinking
    thought_path = agent.think("self", "qualia")

    # Meta-cognition: think about the thought
    meta_concept = agent.meta_think(thought_path)

    # Add more concepts to trigger dimensional expansion
    print("\n🌱 Adding more concepts...")
    agent.add_concept("memory", related_to="thinking", strength=0.7)
    agent.add_concept("emotion", related_to="consciousness", strength=0.8)
    agent.add_concept("reasoning", related_to="thinking", strength=0.8)
    agent.add_concept("intuition", related_to="thinking", strength=0.6)
    agent.add_concept("language", related_to="thinking", strength=0.7)
    agent.add_concept("imagination", related_to="thinking", strength=0.7)

    # Another thought process
    thought_path2 = agent.think("perception", "reasoning")

    # Measure consciousness
    consciousness = agent.measure_consciousness()

    # Visualize
    agent.visualize_conceptual_space(highlight_path=thought_path)

    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)
    print("\n1. SEPARATION → DISTINCTION")
    print("   Concepts need spatial separation to be distinguishable")
    print("   Just like entangled qubits lose distinction!")

    print("\n2. DIMENSIONS → COMPLEXITY")
    print("   More dimensions = room for more distinct concepts")
    print("   Agent automatically expands when space gets crowded")

    print("\n3. META-COGNITION → CONSCIOUSNESS")
    print("   To think about thinking, need SEPARATE space")
    print("   Meta-thoughts exist in different dimensional region")

    print("\n4. WORKING MEMORY → ATTENTION")
    print("   Limited capacity = can only hold ~7 concepts")
    print("   Attention = selecting which dimensions to activate")

    print("\n5. CONSCIOUSNESS ∝ DIMENSIONAL RICHNESS")
    print("   Higher dimensional agents = more consciousness potential")
    print(f"   This agent: {consciousness:.2f} consciousness level")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo_consciousness_architecture()
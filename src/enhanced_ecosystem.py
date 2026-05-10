"""
Enhanced Ecosystem with multiple animal types and complex dynamics

Animals:
- Tiger (apex predator)
- Wolf (pack hunter)
- Deer (herbivore)
- Fox (opportunistic omnivore)
- Rabbit (small herbivore)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .ecosystem import (
    EcosystemEnvironment,
    EcologicalAgent,
    AgentState,
    AgentType,
    DietType,
)


class EnhancedEcologicalAgent(EcologicalAgent):
    """
    Enhanced agent with species-specific traits and behaviors
    """

    def __init__(
        self,
        agent_type: AgentType,
        env: EcosystemEnvironment,
        initial_position: int = 2,
        traits: Dict = None
    ):
        super().__init__(agent_type, env, initial_position)

        # Load species-specific traits
        self.traits = traits or self._get_default_traits(agent_type)

        # Additional state for reproduction and social behavior
        self.reproductive_readiness = np.random.random() < 0.3
        self.pack_size = 1  # Number in social group
        self.home_zone = initial_position  # Preferred home position

    def _get_default_traits(self, agent_type: AgentType) -> Dict:
        """Get default traits for agent type"""
        traits = {
            AgentType.TIGER: {
                "diet": DietType.CARNIVORE,
                "prey": [AgentType.DEER, AgentType.RABBIT, AgentType.FOX],
                "max_hunger_rate": 0.05,
                "max_thirst_rate": 0.04,
                "hunt_success_base": 0.4,
                "energy_per_hunt": 0.3,
                "health_max": 1.0,
                "color": "#ff6b35",
                "speed": 0.7,
                "stealth": 0.8,
                "social": False,
                "reproduction_rate": 0.005,
            },
            AgentType.WOLF: {
                "diet": DietType.CARNIVORE,
                "prey": [AgentType.DEER, AgentType.RABBIT, AgentType.FOX],
                "max_hunger_rate": 0.045,
                "max_thirst_rate": 0.035,
                "hunt_success_base": 0.35,
                "energy_per_hunt": 0.25,
                "health_max": 0.9,
                "color": "#808080",
                "speed": 0.8,
                "stealth": 0.6,
                "social": True,
                "reproduction_rate": 0.008,
            },
            AgentType.DEER: {
                "diet": DietType.HERBIVORE,
                "prey": [],
                "max_hunger_rate": 0.035,
                "max_thirst_rate": 0.045,
                "hunt_success_base": 0.0,
                "energy_per_hunt": 0.0,
                "health_max": 0.85,
                "color": "#8b7355",
                "speed": 0.9,
                "stealth": 0.3,
                "social": True,
                "reproduction_rate": 0.01,
            },
            AgentType.FOX: {
                "diet": DietType.OMNIVORE,
                "prey": [AgentType.RABBIT],
                "max_hunger_rate": 0.03,
                "max_thirst_rate": 0.03,
                "hunt_success_base": 0.3,
                "energy_per_hunt": 0.2,
                "health_max": 0.7,
                "color": "#d2691e",
                "speed": 0.85,
                "stealth": 0.7,
                "social": False,
                "reproduction_rate": 0.01,
            },
            AgentType.RABBIT: {
                "diet": DietType.HERBIVORE,
                "prey": [],
                "max_hunger_rate": 0.025,
                "max_thirst_rate": 0.035,
                "hunt_success_base": 0.0,
                "energy_per_hunt": 0.0,
                "health_max": 0.5,
                "color": "#f5deb3",
                "speed": 0.95,
                "stealth": 0.5,
                "social": False,
                "reproduction_rate": 0.02,  # Rabbits reproduce quickly
            },
        }
        return traits.get(agent_type, {})

    def update_state(self, action: int):
        """Update internal state with species-specific traits"""
        # Call parent update first
        super().update_state(action)

        # Apply species-specific hunger/thirst rates
        self.state.hunger = min(1.0, self.state.hunger + self.traits["max_hunger_rate"])
        self.state.thirst = min(1.0, self.state.thirst + self.traits["max_thirst_rate"])

        # Check for reproduction
        if self.reproductive_readiness and self.state.health > 0.7 and self.state.age > 50:
            if np.random.random() < self.traits["reproduction_rate"]:
                # Could trigger creation of new agent
                self.reproductive_readiness = False

    def get_hunt_success_probability(self, prey_agent: 'EnhancedEcologicalAgent') -> float:
        """Calculate probability of successful hunt based on traits"""
        base = self.traits["hunt_success_base"]

        # Modifiers
        stealth_mod = self.traits.get("stealth", 0.5) * 0.2
        speed_mod = self.traits.get("speed", 0.5) * 0.15
        health_mod = self.state.health * 0.1
        energy_mod = self.state.energy * 0.1

        # Prey defense modifiers
        prey_speed = prey_agent.traits.get("speed", 0.5)
        prey_stealth = prey_agent.traits.get("stealth", 0.5)

        success_prob = base + stealth_mod + speed_mod + health_mod + energy_mod
        success_prob -= prey_speed * 0.15
        success_prob -= prey_stealth * 0.1

        # Environmental factors
        visibility = self.env.get_visibility()
        success_prob *= visibility

        return max(0.05, min(0.9, success_prob))


class EnhancedEcosystemSimulator:
    """
    Enhanced ecosystem simulation with all animal types,
    complex interactions, and emergent behaviors.
    """

    def __init__(
        self,
        n_tigers: int = 1,
        n_wolves: int = 2,
        n_deer: int = 5,
        n_foxes: int = 3,
        n_rabbits: int = 8
    ):
        self.env = EcosystemEnvironment()
        self.agents: List[EnhancedEcologicalAgent] = []
        self.step_count = 0

        # Create agents by type
        agent_configs = [
            (AgentType.TIGER, n_tigers, 2),    # Tigers start in safe zone
            (AgentType.WOLF, n_wolves, 0),     # Wolves start in prey zone
            (AgentType.DEER, n_deer, 3),       # Deer start in grass zone
            (AgentType.FOX, n_foxes, 2),       # Foxes start in safe zone
            (AgentType.RABBIT, n_rabbits, 3),  # Rabbits start in grass zone
        ]

        for agent_type, count, start_pos in agent_configs:
            for _ in range(count):
                agent = EnhancedEcologicalAgent(agent_type, self.env, start_pos)
                self.agents.append(agent)

        self.history: List[Dict] = []

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one step"""
        self.env.step()
        self.step_count += 1

        # Process each alive agent
        for agent in self.agents:
            if not agent.is_alive():
                continue

            # Detect nearby agents
            nearby_agents = self._get_nearby_agents(agent)

            # Perceive and act
            obs = agent.perceive()
            action, _ = agent.act(obs)
            agent.update_state(action)

        # Resolve interactions
        self._resolve_interactions()

        # Remove dead agents
        self.agents = [a for a in self.agents if a.is_alive()]

        # Record statistics
        stats = self._collect_statistics()
        self.history.append(stats)

        return stats

    def _get_nearby_agents(self, agent: EnhancedEcologicalAgent) -> List[EnhancedEcologicalAgent]:
        """Get agents in the same position with visibility consideration"""
        nearby = []
        visibility = self.env.get_visibility()

        for other in self.agents:
            if other is agent or not other.is_alive():
                continue

            if other.state.position == agent.state.position:
                # Detection probability based on visibility and agent stealth
                detection_prob = visibility * (1 - other.traits.get("stealth", 0.5))
                if np.random.random() < detection_prob:
                    nearby.append(other)

        return nearby

    def _resolve_interactions(self):
        """Resolve all agent interactions (predation, competition, etc.)"""
        # Group agents by position
        agents_by_position = {}
        for agent in self.agents:
            if not agent.is_alive():
                continue
            pos = agent.state.position
            if pos not in agents_by_position:
                agents_by_position[pos] = []
            agents_by_position[pos].append(agent)

        # Process each position
        for position, agents in agents_by_position.items():
            if len(agents) <= 1:
                continue

            # Separate predators and prey
            predators = [a for a in agents if a.traits["diet"] in [DietType.CARNIVORE, DietType.OMNIVORE]]
            prey = [a for a in agents if a.traits["diet"] == DietType.HERBIVORE]

            # Predation
            for predator in predators:
                if not predator.is_alive():
                    continue

                for prey_agent in prey:
                    if not prey_agent.is_alive():
                        continue

                    # Check if prey is in predator's diet
                    if prey_agent.agent_type not in predator.traits.get("prey", []):
                        continue

                    # Calculate hunt success
                    success_prob = predator.get_hunt_success_probability(prey_agent)

                    if np.random.random() < success_prob:
                        # Successful hunt!
                        prey_agent.state.alive = False
                        predator.state.hunger = max(0, predator.state.hunger - 0.8)
                        predator.state.health = min(1.0, predator.state.health + 0.15)
                        break  # Predator can only catch one prey per step

    def _collect_statistics(self) -> Dict[str, Any]:
        """Collect statistics for current step"""
        stats = {
            'step': self.step_count,
            'time': self.env.time_step,
            'day_time': self.env.day_time,
            'weather': self.env.weather,
            'alive': sum(1 for a in self.agents if a.is_alive()),
        }

        # Count by species
        for agent_type in AgentType:
            count = sum(1 for a in self.agents if a.agent_type == agent_type and a.is_alive())
            stats[f'{agent_type.value}_alive'] = count

        # Resource levels
        stats['resources'] = {k: v.amount for k, v in self.env.resources.items()}

        # Average health by species
        for agent_type in AgentType:
            agents_of_type = [a for a in self.agents if a.agent_type == agent_type and a.is_alive()]
            if agents_of_type:
                avg_health = np.mean([a.state.health for a in agents_of_type])
                stats[f'{agent_type.value}_avg_health'] = round(avg_health, 2)

        return stats

    def run(self, max_steps: int = 500) -> List[Dict]:
        """Run simulation for specified number of steps"""
        for _ in range(max_steps):
            stats = self.step()

            # Stop if no animals left
            if stats['alive'] == 0:
                break

        return self.history

    def get_summary(self) -> str:
        """Get human-readable summary of simulation"""
        if not self.history:
            return "No simulation data"

        final = self.history[-1]
        summary = "Enhanced Ecosystem Simulation Summary\n"
        summary += "===================================\n"
        summary += f"Total Steps: {final['step']}\n"
        summary += f"Total Animals Alive: {final['alive']}\n"
        summary += f"Time of Day: {'Day' if final['day_time'] else 'Night'}\n"
        summary += f"Weather: {final['weather']}\n\n"
        summary += "Population by Species:\n"

        for agent_type in AgentType:
            key = f'{agent_type.value}_alive'
            if key in final:
                summary += f"  {agent_type.value.capitalize()}: {final[key]}\n"

        return summary


def run_demo():
    """Run a demonstration of the enhanced ecosystem"""
    sim = EnhancedEcosystemSimulator(
        n_tigers=1,
        n_wolves=2,
        n_deer=5,
        n_foxes=3,
        n_rabbits=8
    )

    print("🌍 Running Enhanced Ecosystem Simulation")
    print("Species:")
    print("  🐅 Tiger (apex predator)")
    print("  🐺 Wolf (pack hunter)")
    print("  🦌 Deer (herbivore)")
    print("  🦊 Fox (opportunistic)")
    print("  🐇 Rabbit (fast breeder)")

    history = sim.run(max_steps=300)

    print("\n" + "=" * 50)
    print(sim.get_summary())

    return sim


if __name__ == "__main__":
    run_demo()

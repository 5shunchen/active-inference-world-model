"""
Multi-Agent Ecosystem Simulation
Implements predator-prey dynamics with active inference agents.

Agents:
- Tiger (predator): hunts prey, needs food/water/safety
- Deer (prey): grazes, avoids predators, needs food/water
- Environment: dynamic resources, weather, day/night cycle
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .active_inference import (
    GenerativeModelAdvanced,
    ActiveInferenceAgentAdvanced,
    MarkovDecisionProcess
)


class AgentType(Enum):
    TIGER = "tiger"
    DEER = "deer"


class Position(Enum):
    PREY_ZONE = 0
    WATER_ZONE = 1
    SAFE_ZONE = 2
    GRASS_ZONE = 3


class Action(Enum):
    MOVE_TO_PREY = 0
    MOVE_TO_WATER = 1
    MOVE_TO_SAFE = 2
    MOVE_TO_GRASS = 3
    EAT = 4
    DRINK = 5
    REST = 6
    HUNT = 7
    FLEE = 8


@dataclass
class AgentState:
    """Internal state of an ecological agent"""
    position: int
    hunger: float  # 0 = full, 1 = starving
    thirst: float  # 0 = hydrated, 1 = dehydrated
    health: float  # 0 = dead, 1 = perfect health
    energy: float  # 0 = exhausted, 1 = full energy
    age: float     # Age in simulation steps
    alive: bool = True


@dataclass
class Resource:
    """Environmental resource (food, water)"""
    position: int
    amount: float
    regeneration_rate: float = 0.05


class EcosystemEnvironment:
    """
    Dynamic ecosystem environment with:
    - Multiple resource locations
    - Day/night cycle
    - Weather effects
    - Multi-agent interactions
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.time_step = 0
        self.day_time = True  # Day/night cycle
        self.weather = "clear"  # clear, rain, fog

        # Resources
        self.resources = {
            'water': Resource(position=1, amount=1.0),
            'grass': Resource(position=3, amount=0.8),
            'prey_spawn': Resource(position=0, amount=0.5),
        }

        # Environment effects matrix
        # position effects: [safety, food, water, visibility]
        self.position_effects = np.array([
            [0.3, 0.8, 0.1, 0.7],  # PREY_ZONE
            [0.5, 0.1, 0.9, 0.6],  # WATER_ZONE
            [0.9, 0.1, 0.1, 0.8],  # SAFE_ZONE
            [0.4, 0.9, 0.3, 0.7],  # GRASS_ZONE
        ])

    def step(self):
        """Advance environment by one time step"""
        self.time_step += 1

        # Day/night cycle (every 50 steps)
        if self.time_step % 50 == 0:
            self.day_time = not self.day_time

        # Weather changes
        if np.random.random() < 0.02:
            weather_options = ['clear', 'cloudy', 'rain', 'fog']
            self.weather = np.random.choice(weather_options)

        # Regenerate resources
        for resource in self.resources.values():
            resource.amount = min(1.0, resource.amount + resource.regeneration_rate)

    def get_visibility(self) -> float:
        """Get current visibility based on weather and time"""
        base = 0.8 if self.day_time else 0.4
        if self.weather == 'fog':
            base *= 0.5
        elif self.weather == 'rain':
            base *= 0.7
        return base

    def get_safety(self, position: int) -> float:
        """Get safety level at position"""
        safety = self.position_effects[position, 0]
        if not self.day_time:
            safety *= 0.7  # Night is less safe
        return safety

    def consume_resource(self, resource_name: str, amount: float) -> float:
        """Consume a resource, returns amount actually consumed"""
        if resource_name in self.resources:
            consumed = min(amount, self.resources[resource_name].amount)
            self.resources[resource_name].amount -= consumed
            return consumed
        return 0.0


class EcologicalAgent:
    """
    Active inference agent embedded in an ecosystem.

    Each agent has its own generative model and preferences
    based on its ecological niche (predator/prey).
    """

    def __init__(
        self,
        agent_type: AgentType,
        env: EcosystemEnvironment,
        initial_position: int = 2
    ):
        self.agent_type = agent_type
        self.env = env

        # Internal state
        self.state = AgentState(
            position=initial_position,
            hunger=np.random.uniform(0.3, 0.5),
            thirst=np.random.uniform(0.2, 0.4),
            health=1.0,
            energy=1.0,
            age=0.0
        )

        # Set up active inference model
        self._setup_generative_model()

    def _setup_generative_model(self):
        """Create agent-specific generative model"""
        n_states = 4  # positions
        n_observations = 8  # hunger, thirst, safety, food, water, predator, prey, time
        n_actions = 6  # move to each zone, eat, drink, rest, hunt, flee

        if self.agent_type == AgentType.TIGER:
            # Tiger prefers hunting prey, high health, safety
            self.model = GenerativeModelAdvanced(n_states, n_observations, n_actions)

            # Likelihood: noisy position -> observation mapping
            A = np.eye(n_observations, n_states) * 0.7 + 0.3 / n_states
            self.model.set_likelihood(A)

            # Transition: actions move agent to different zones
            B = np.zeros((n_states, n_states, n_actions))
            for a in range(min(n_actions, 4)):
                B[a, :, a] = 0.8  # Action a moves to state a with 80% prob
                for s in range(n_states):
                    B[:, s, a] += 0.05  # Noise
                    B[:, s, a] /= B[:, s, a].sum()

            # Remaining actions (eat, drink) don't change position
            for a in range(4, n_actions):
                for s in range(n_states):
                    B[s, s, a] = 1.0

            self.model.set_transition(B)

            # Tiger preferences: prey_zone, health, low hunger
            preferences = np.zeros(n_observations)
            preferences[0] = 2.0  # Prefer being in prey zone for hunting
            preferences[3] = 1.5  # Food available
            preferences[6] = 3.0  # Prey detected
            self.model.set_preferences(preferences)

        else:  # DEER
            # Deer prefers safety, food, avoiding predators
            self.model = GenerativeModelAdvanced(n_states, n_observations, n_actions)

            A = np.eye(n_observations, n_states) * 0.6 + 0.4 / n_states
            self.model.set_likelihood(A)

            B = np.zeros((n_states, n_states, n_actions))
            for a in range(min(n_actions, 4)):
                B[a, :, a] = 0.85  # Deer move faster
                for s in range(n_states):
                    B[:, s, a] += 0.0375
                    B[:, s, a] /= B[:, s, a].sum()

            for a in range(4, n_actions):
                for s in range(n_states):
                    B[s, s, a] = 1.0

            self.model.set_transition(B)

            # Deer preferences: safety, grass, water, avoiding tiger
            preferences = np.zeros(n_observations)
            preferences[2] = 5.0  # Safety is paramount
            preferences[3] = 2.0  # Food available
            preferences[4] = 2.0  # Water available
            preferences[5] = -10.0  # Strong aversion to predators
            self.model.set_preferences(preferences)

        self.agent = ActiveInferenceAgentAdvanced(self.model)

    def perceive(self) -> int:
        """Generate observation from environment and internal state"""
        # Composite observation based on state and environment
        obs_components = [
            self.state.position,  # Position
            int(self.state.hunger > 0.7),  # High hunger
            int(self.env.get_safety(self.state.position) < 0.5),  # Unsafe
            int(self.env.resources['grass'].amount > 0.5 if self.state.position == 3 else
                self.env.resources['prey_spawn'].amount > 0.3 if self.state.position == 0 else 0),  # Food available
            int(self.state.position == 1 and self.env.resources['water'].amount > 0.3),  # Water available
            0,  # Predator detected (set by ecosystem)
            0,  # Prey detected (set by ecosystem)
            int(self.env.day_time),  # Day/night
        ]

        # Simplify to single observation (composite encoding)
        obs = hash(tuple(obs_components)) % self.model.n_observations
        return obs

    def act(self, observation: int) -> Tuple[int, float]:
        """Execute perception-action cycle, returns (action, VFE)"""
        action, vfe, efe = self.agent.step(observation)
        return action, vfe

    def update_state(self, action: int):
        """Update internal state based on action and environment"""
        # Move if action is movement
        if action < 4:
            self.state.position = action

        # Eat action (only effective in food zones)
        elif action == 4:
            if self.state.position in [0, 3]:  # Prey or grass zone
                resource = 'prey_spawn' if self.state.position == 0 else 'grass'
                consumed = self.env.consume_resource(resource, 0.3)
                if consumed > 0:
                    self.state.hunger = max(0, self.state.hunger - consumed)
                    self.state.energy = min(1.0, self.state.energy + consumed * 0.5)

        # Drink action
        elif action == 5:
            if self.state.position == 1:  # Water zone
                consumed = self.env.consume_resource('water', 0.4)
                if consumed > 0:
                    self.state.thirst = max(0, self.state.thirst - consumed)

        # Rest action
        elif action == 6:
            self.state.energy = min(1.0, self.state.energy + 0.2)

        # Energy cost for all actions
        energy_cost = 0.05 if action < 4 else 0.02  # Movement costs more
        self.state.energy = max(0, self.state.energy - energy_cost)

        # Hunger and thirst increase over time
        self.state.hunger = min(1.0, self.state.hunger + 0.03)
        self.state.thirst = min(1.0, self.state.thirst + 0.04)

        # Age increases
        self.state.age += 1

        # Health degradation if needs not met
        if self.state.hunger > 0.9 or self.state.thirst > 0.9:
            self.state.health -= 0.05
        elif self.state.hunger < 0.3 and self.state.thirst < 0.3:
            self.state.health = min(1.0, self.state.health + 0.01)

        # Check death
        if self.state.health <= 0:
            self.state.alive = False

    def is_alive(self) -> bool:
        return self.state.alive


class EcosystemSimulator:
    """
    Full ecosystem simulation with multiple agents.

    Manages predator-prey dynamics, resource competition,
    and environment-agent interactions.
    """

    def __init__(self, n_tigers: int = 1, n_deer: int = 3):
        self.env = EcosystemEnvironment()
        self.agents: List[EcologicalAgent] = []

        # Create tigers
        for i in range(n_tigers):
            tiger = EcologicalAgent(AgentType.TIGER, self.env, initial_position=2)
            self.agents.append(tiger)

        # Create deer
        for i in range(n_deer):
            deer = EcologicalAgent(AgentType.DEER, self.env, initial_position=3)
            self.agents.append(deer)

        self.history: List[Dict] = []

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one step"""
        self.env.step()

        step_stats = {
            'time': self.env.time_step,
            'day_time': self.env.day_time,
            'weather': self.env.weather,
            'alive': sum(1 for a in self.agents if a.is_alive()),
            'tigers_alive': sum(1 for a in self.agents if a.agent_type == AgentType.TIGER and a.is_alive()),
            'deer_alive': sum(1 for a in self.agents if a.agent_type == AgentType.DEER and a.is_alive()),
            'resources': {k: v.amount for k, v in self.env.resources.items()},
        }

        # Agent actions
        for agent in self.agents:
            if not agent.is_alive():
                continue

            # Detect nearby agents for predator/prey awareness
            self._update_detection(agent)

            # Perceive and act
            obs = agent.perceive()
            action, vfe = agent.act(obs)
            agent.update_state(action)

        # Handle predation events
        self._resolve_predation()

        self.history.append(step_stats)
        return step_stats

    def _update_detection(self, agent: EcologicalAgent):
        """Update agent's awareness of other agents"""
        visibility = self.env.get_visibility()

        for other in self.agents:
            if other is agent or not other.is_alive():
                continue

            # Same position = detection
            if other.state.position == agent.state.position and np.random.random() < visibility:
                if agent.agent_type == AgentType.DEER and other.agent_type == AgentType.TIGER:
                    # Deer detected tiger - modify observation
                    pass  # This would feed into observation generation

    def _resolve_predation(self):
        """Handle predation events when predators and prey occupy same space"""
        tigers = [a for a in self.agents if a.agent_type == AgentType.TIGER and a.is_alive()]
        deer = [a for a in self.agents if a.agent_type == AgentType.DEER and a.is_alive()]

        for tiger in tigers:
            for d in deer:
                if d.is_alive() and tiger.state.position == d.state.position:
                    # Hunting success depends on tiger energy, health, and visibility
                    success_chance = (
                        0.3 +
                        0.3 * tiger.state.energy +
                        0.2 * tiger.state.health +
                        0.2 * self.env.get_visibility()
                    )

                    if np.random.random() < success_chance:
                        # Successful hunt!
                        d.state.alive = False
                        tiger.state.hunger = max(0, tiger.state.hunger - 0.8)
                        tiger.state.health = min(1.0, tiger.state.health + 0.2)
                        break

    def run(self, max_steps: int = 500) -> List[Dict]:
        """Run simulation for specified number of steps"""
        for _ in range(max_steps):
            stats = self.step()
            if stats['alive'] == 0:
                break

        return self.history

    def get_summary(self) -> str:
        """Get human-readable summary of simulation"""
        if not self.history:
            return "No simulation data"

        final = self.history[-1]
        return (
            f"Ecosystem Simulation Summary\n"
            f"============================\n"
            f"Total Steps: {final['time']}\n"
            f"Surviving Tigers: {final['tigers_alive']}\n"
            f"Surviving Deer: {final['deer_alive']}\n"
            f"Final Weather: {final['weather']}\n"
            f"Time of Day: {'Day' if final['day_time'] else 'Night'}\n"
        )

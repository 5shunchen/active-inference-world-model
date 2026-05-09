"""
Stage 1: World Simulator
Hides true state from agent, only exposes step(action) -> observation and observe()
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any

from .generative_model import GenerativeModel


class Trajectory:
    """Records a full simulation trajectory"""

    def __init__(self):
        self.states: List[Any] = []
        self.observations: List[Any] = []
        self.actions: List[Any] = []
        self.free_energy: List[float] = []
        self.beliefs: List[np.ndarray] = []
        self.timesteps: List[int] = []

    def record(self, timestep: int, state: Any, observation: Any, action: Any = None,
               free_energy: float = None, belief: np.ndarray = None):
        """Record a single timestep"""
        self.timesteps.append(timestep)
        self.states.append(state)
        self.observations.append(observation)
        if action is not None:
            self.actions.append(action)
        if free_energy is not None:
            self.free_energy.append(free_energy)
        if belief is not None:
            self.beliefs.append(belief)


class WorldSimulator:
    """
    World simulator that hides the true state from the agent.
    Only exposes observations through step(action) and observe()
    """

    def __init__(self, generative_model: GenerativeModel, initial_state: Optional[int] = None):
        self.model = generative_model
        self._true_state: int = initial_state if initial_state is not None else self._sample_initial_state()
        self.timestep = 0
        self.trajectory = Trajectory()
        # Record initial state and observation
        initial_obs = self.observe()
        self.trajectory.record(self.timestep, self._true_state, initial_obs)

    def _sample_initial_state(self) -> int:
        """Sample initial state from the model's initial belief"""
        return self.model.get_initial_belief().sample()[0]

    def observe(self) -> int:
        """Get an observation of the current true state"""
        obs_model = self.model.get_observation_model()
        return obs_model.sample_observation(self._true_state)

    def step(self, action: int) -> Tuple[int, bool]:
        """
        Execute an action in the world and return the next observation.
        Returns: (observation, done)
        """
        # Transition to next state using true dynamics
        self._true_state = self.model.step_dynamics(self._true_state, action)
        self.timestep += 1

        # Sample observation
        observation = self.observe()

        # Record to trajectory (action was taken to get here)
        self.trajectory.record(self.timestep, self._true_state, observation, action=action)

        # Check termination (for now, no termination)
        done = False

        return observation, done

    def get_true_state(self) -> int:
        """For debugging/testing only - should not be used by agent"""
        return self._true_state

    def reset(self) -> int:
        """Reset the world to initial state"""
        self._true_state = self._sample_initial_state()
        self.timestep = 0
        self.trajectory = Trajectory()
        initial_obs = self.observe()
        self.trajectory.record(self.timestep, self._true_state, initial_obs)
        return initial_obs

    def __repr__(self):
        return f"WorldSimulator(state={self._true_state}, t={self.timestep})"

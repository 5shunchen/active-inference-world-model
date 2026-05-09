"""
Stage 0: Generative Model Abstract Base Class
Defines the core interface for all generative models in active inference
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union

from .distributions import (
    Distribution,
    Categorical,
    MultivariateGaussian,
    LikelihoodModel,
    TransitionModel
)


class GenerativeModel(ABC):
    """
    Abstract base class for generative models in active inference.

    A generative model defines P(o, s, a) = P(o|s) P(s|s_prev, a_prev) P(a)
    """

    def __init__(self, n_states: int, n_observations: int, n_actions: int):
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions

    @abstractmethod
    def get_observation_model(self) -> LikelihoodModel:
        """Return the observation likelihood model P(o|s)"""
        pass

    @abstractmethod
    def get_transition_model(self) -> TransitionModel:
        """Return the state transition model P(s'|s, a)"""
        pass

    @abstractmethod
    def get_preferences(self, state: Optional[int] = None) -> np.ndarray:
        """
        Return the prior preferences C matrix (log probabilities) over observations/states.
        This drives action selection to minimize expected free energy.
        """
        pass

    @abstractmethod
    def get_initial_belief(self) -> Distribution:
        """Return the initial belief distribution over states"""
        pass

    @abstractmethod
    def step_dynamics(self, state: Union[int, np.ndarray], action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Execute one step of the true environment dynamics (for WorldSimulator use).
        Returns the next true state.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(n_states={self.n_states}, n_obs={self.n_observations}, n_actions={self.n_actions})"


class DiscreteGenerativeModel(GenerativeModel):
    """
    Concrete implementation for discrete state spaces"""

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        n_actions: int,
        observation_matrix: np.ndarray,
        transition_tensor: np.ndarray,
        preferences: np.ndarray,
        initial_state_dist: np.ndarray
    ):
        super().__init__(n_states, n_observations, n_actions)

        self._observation_model = LikelihoodModel(observation_matrix)
        self._transition_model = TransitionModel(transition_tensor)
        self._preferences = preferences
        self._initial_belief = Categorical(initial_state_dist)

    def get_observation_model(self) -> LikelihoodModel:
        return self._observation_model

    def get_transition_model(self) -> TransitionModel:
        return self._transition_model

    def get_preferences(self, state: Optional[int] = None) -> np.ndarray:
        return self._preferences

    def get_initial_belief(self) -> Categorical:
        return self._initial_belief

    def step_dynamics(self, state: int, action: int) -> int:
        return self._transition_model.sample_next_state(state, action)


class ContinuousGenerativeModel(GenerativeModel):
    """
    Concrete implementation for continuous state spaces (placeholder)
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        transition_noise_cov: np.ndarray,
        observation_noise_cov: np.ndarray
    ):
        super().__init__(state_dim, obs_dim, action_dim)
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.transition_noise_cov = transition_noise_cov
        self.observation_noise_cov = observation_noise_cov

    def get_observation_model(self):
        # Linear observation model for continuous spaces
        raise NotImplementedError("Continuous observation model to be implemented")

    def get_transition_model(self):
        # Linear transition model for continuous spaces
        raise NotImplementedError("Continuous transition model to be implemented")

    def get_preferences(self, state=None):
        raise NotImplementedError("Continuous preferences to be implemented")

    def get_initial_belief(self) -> MultivariateGaussian:
        return MultivariateGaussian(np.zeros(self.state_dim), np.eye(self.state_dim))

    def step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # Simple linear dynamics: x' = Ax + Bu + noise
        raise NotImplementedError("Continuous dynamics to be implemented")

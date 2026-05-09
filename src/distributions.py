"""
Stage 0: Probability Distributions
Implements Categorical and MultivariateGaussian distributions with sample() and log_prob()
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class Distribution(ABC):
    """Abstract base class for all probability distributions"""

    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n samples from the distribution"""
        pass

    @abstractmethod
    def log_prob(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """Compute log probability of x under this distribution"""
        pass


class Categorical(Distribution):
    """Categorical distribution over discrete states"""

    def __init__(self, probabilities: Union[list, np.ndarray]):
        self.probs = np.asarray(probabilities, dtype=np.float64)
        # Normalize to ensure sum = 1
        self.probs = self.probs / np.sum(self.probs)
        self.n_states = len(self.probs)

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n samples (returns state indices)"""
        return np.random.choice(self.n_states, size=n, p=self.probs)

    def log_prob(self, x: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability of state(s) x"""
        x = np.asarray(x, dtype=int)
        return np.log(self.probs[x] + 1e-12)  # +epsilon to avoid log(0)

    def __repr__(self):
        return f"Categorical(probs={self.probs})"


class MultivariateGaussian(Distribution):
    """Multivariate Gaussian (Normal) distribution"""

    def __init__(self, mean: Union[list, np.ndarray], cov: Union[list, np.ndarray]):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.dim = len(self.mean)

        # Ensure covariance is properly shaped
        if self.cov.shape != (self.dim, self.dim):
            raise ValueError(f"Covariance shape {self.cov.shape} does not match mean dimension {self.dim}")

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n samples from multivariate Gaussian"""
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def log_prob(self, x: Union[list, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability of x under multivariate Gaussian"""
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov + 1e-6 * np.eye(self.dim))
        det_cov = np.linalg.det(self.cov + 1e-6 * np.eye(self.dim))

        norm_const = -0.5 * (self.dim * np.log(2 * np.pi) + np.log(det_cov))
        exponent = -0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)

        result = norm_const + exponent
        return result[0] if len(result) == 1 else result

    def __repr__(self):
        return f"MultivariateGaussian(mean={self.mean}, cov_shape={self.cov.shape})"


class LikelihoodModel:
    """Observation likelihood model: P(observation | state)"""

    def __init__(self, matrix: np.ndarray):
        """
        Args:
            matrix: Shape (n_observations, n_states) where matrix[o, s] = P(o|s)
        """
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.n_observations, self.n_states = self.matrix.shape

    def get_observation_prob(self, state: int) -> np.ndarray:
        """Get P(observation | state) for a given state"""
        return self.matrix[:, state]

    def sample_observation(self, state: int) -> int:
        """Sample an observation given a state"""
        return Categorical(self.get_observation_prob(state)).sample()[0]

    def __repr__(self):
        return f"LikelihoodModel(shape={self.matrix.shape})"


class TransitionModel:
    """State transition model: P(next_state | current_state, action)"""

    def __init__(self, tensor: np.ndarray):
        """
        Args:
            tensor: Shape (n_states, n_states, n_actions) where tensor[s', s, a] = P(s'|s, a)
        """
        self.tensor = np.asarray(tensor, dtype=np.float64)
        self.n_states_next, self.n_states, self.n_actions = self.tensor.shape

        assert self.n_states_next == self.n_states, "Transition tensor must be square in states"

    def get_transition_prob(self, state: int, action: int) -> np.ndarray:
        """Get P(next_state | current_state, action)"""
        return self.tensor[:, state, action]

    def sample_next_state(self, state: int, action: int) -> int:
        """Sample next state given current state and action"""
        probs = self.get_transition_prob(state, action)
        return Categorical(probs).sample()[0]

    def __repr__(self):
        return f"TransitionModel(shape={self.tensor.shape})"

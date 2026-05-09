"""
Stage 1: Inference Engine for Active Inference
Implements: infer_state(), compute_efe(), plan_action()
"""

import numpy as np
from abc import ABC, abstractmethod

from .generative_model import GenerativeModel


class InferenceEngine(ABC):
    """Abstract base class for inference engines"""

    def __init__(self, model: GenerativeModel):
        self.model = model
        self.current_belief: np.ndarray = model.get_initial_belief().probs.copy()

    @abstractmethod
    def infer_state(self, observation: int) -> np.ndarray:
        """Update belief state given an observation (Bayesian filtering)"""
        pass

    @abstractmethod
    def compute_efe(self, initial_belief: np.ndarray, action: int, horizon: int = 1) -> float:
        """Compute Expected Free Energy (EFE) for a given action"""
        pass

    @abstractmethod
    def plan_action(self, horizon: int = 1) -> int:
        """Plan the next action to take based on EFE minimization"""
        pass

    def get_free_energy(self, observation: int) -> float:
        """Compute variational free energy for current belief and observation"""
        # Free energy F = E_q[ln q(s) - ln P(o|s) - ln P(s)]
        # For discrete case: F = sum_s q(s) * [ln q(s) - ln P(o|s) - ln P(s)]
        qs = self.current_belief
        A = self.model.get_observation_model().matrix
        ln_Po_given_s = np.log(A[observation, :] + 1e-12)

        # Use uniform prior for simplicity
        ln_Ps = np.log(np.ones_like(qs) / len(qs))

        entropy = -np.sum(qs * np.log(qs + 1e-12))
        energy = -np.sum(qs * (ln_Po_given_s + ln_Ps))

        return entropy + energy


class DiscreteInferenceEngine(InferenceEngine):
    """
    Discrete state inference engine using exact Bayesian filtering.
    Policy selection via brute-force EFE evaluation.
    """

    def __init__(self, model: GenerativeModel):
        super().__init__(model)

    def infer_state(self, observation: int) -> np.ndarray:
        """
        Bayesian filtering: q(s) ∝ P(o|s) * q_prev(s)
        """
        A = self.model.get_observation_model().matrix
        likelihood = A[observation, :]

        # Update belief: posterior ∝ likelihood * prior
        unnormalized = likelihood * self.current_belief
        self.current_belief = unnormalized / (np.sum(unnormalized) + 1e-12)

        return self.current_belief.copy()

    def compute_efe(self, initial_belief: np.ndarray, action: int, horizon: int = 1) -> float:
        """
        Compute Expected Free Energy for taking a specific action from current belief
        EFE = Risk + Ambiguity

        Risk: Expected utility term -E_q[ln P(o)] = divergence from preferred observations
        Ambiguity: -E_q[H[P(o|s)]]
        """
        # Get transition model for this action
        B = self.model.get_transition_model().tensor
        B_action = B[:, :, action]  # P(s'|s, a)

        # Expected next state: q(s') = sum_s P(s'|s, a) * q(s)
        expected_next_state = B_action @ initial_belief

        # Expected observations: q(o) = sum_{s'} P(o|s') * q(s')
        A = self.model.get_observation_model().matrix
        expected_obs = A @ expected_next_state

        # Preferences over observations (log probabilities)
        C = self.model.get_preferences()

        # Risk term: KL divergence between expected observations and preferences
        # Risk = sum_o expected_obs[o] * (ln expected_obs[o] - C[o])
        risk = np.sum(expected_obs * (np.log(expected_obs + 1e-12) - C))

        # Ambiguity term: Expected entropy of observation likelihood
        # H[P(o|s')] = -sum_o P(o|s') * ln P(o|s')
        entropy_per_state = -np.sum(A * np.log(A + 1e-12), axis=0)
        ambiguity = np.sum(expected_next_state * entropy_per_state)

        return risk + ambiguity

    def plan_action(self, horizon: int = 1) -> int:
        """
        Select action that minimizes Expected Free Energy.
        Uses brute-force evaluation over all possible actions.
        """
        efe_values = np.zeros(self.model.n_actions)

        for action in range(self.model.n_actions):
            efe_values[action] = self.compute_efe(self.current_belief, action, horizon)

        # Select action with minimal EFE
        return int(np.argmin(efe_values))


class ActiveInferenceAgent:
    """
    Complete active inference agent combining perception and action.
    """

    def __init__(self, model: GenerativeModel):
        self.inference_engine = DiscreteInferenceEngine(model)

    def perceive(self, observation: int) -> float:
        """Update belief based on observation, return free energy"""
        self.inference_engine.infer_state(observation)
        return self.inference_engine.get_free_energy(observation)

    def act(self, horizon: int = 1) -> int:
        """Choose next action based on EFE minimization"""
        return self.inference_engine.plan_action(horizon)

    @property
    def belief(self) -> np.ndarray:
        return self.inference_engine.current_belief

    @property
    def model(self) -> GenerativeModel:
        return self.inference_engine.model


def run_active_inference_loop(
    agent: ActiveInferenceAgent,
    world,
    n_steps: int = 100,
    horizon: int = 1
) -> None:
    """
    Run the active inference loop: perceive -> act -> repeat
    """
    for _ in range(n_steps):
        # Get current observation
        obs = world.observe()

        # Perceive: update belief
        fe = agent.perceive(obs)

        # Act: select next action
        action = agent.act(horizon)

        # Step the world
        world.step(action)

        # Record free energy in trajectory
        if world.trajectory.free_energy:
            world.trajectory.free_energy[-1] = fe
        else:
            world.trajectory.free_energy.append(fe)

        # Record belief
        world.trajectory.beliefs.append(agent.belief.copy())

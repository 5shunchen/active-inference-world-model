"""
Enhanced Active Inference Engine
Implements proper Expected Free Energy (EFE) minimization
with prior preferences, ambiguity, and risk terms.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class GenerativeModelAdvanced:
    """Advanced generative model with full active inference components"""

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        n_actions: int,
        horizon: int = 3
    ):
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.horizon = horizon

        # Initialize distributions
        # A: P(o | s) - Likelihood matrix
        self.A = np.ones((n_observations, n_states)) / n_observations

        # B: P(s' | s, a) - Transition tensor
        self.B = np.ones((n_states, n_states, n_actions)) / n_states

        # C: ln P(o) - Log prior preferences over observations
        self.C = np.zeros(n_observations)

        # D: P(s_0) - Initial state prior
        self.D = np.ones(n_states) / n_states

    def set_likelihood(self, A: np.ndarray):
        """Set observation likelihood matrix"""
        assert A.shape == (self.n_observations, self.n_states)
        self.A = A / A.sum(axis=0, keepdims=True)

    def set_transition(self, B: np.ndarray):
        """Set transition tensor"""
        assert B.shape == (self.n_states, self.n_states, self.n_actions)
        for a in range(self.n_actions):
            self.B[:, :, a] = B[:, :, a] / B[:, :, a].sum(axis=0, keepdims=True)

    def set_preferences(self, preferences: np.ndarray):
        """Set log preferences over observations"""
        assert preferences.shape == (self.n_observations,)
        # Normalize to proper log probabilities
        self.C = preferences - np.log(np.sum(np.exp(preferences)))

    def set_initial_prior(self, D: np.ndarray):
        """Set initial state distribution"""
        assert D.shape == (self.n_states,)
        self.D = D / D.sum()


class ActiveInferenceAgentAdvanced:
    """
    Advanced Active Inference Agent with proper EFE computation.

    Implements:
    - Variational filtering for state inference
    - Expected Free Energy (EFE) for action planning
    - Policy selection via EFE minimization
    """

    def __init__(self, model: GenerativeModelAdvanced):
        self.model = model
        self.q_s = model.D.copy()  # Current variational belief over states
        self.posterior_history: List[np.ndarray] = [self.q_s.copy()]
        self.efe_history: List[np.ndarray] = []

    def infer_states(self, observation: int) -> float:
        """
        Perception: Update state beliefs based on observation using
        variational Bayes filtering.

        Returns: Variational Free Energy (surprise)
        """
        # q(s) ∝ P(o | s) * q_prev(s)
        likelihood = self.model.A[observation, :]
        unnormalized = likelihood * self.q_s
        self.q_s = unnormalized / (unnormalized.sum() + 1e-12)
        self.posterior_history.append(self.q_s.copy())

        # Compute Variational Free Energy (VFE)
        # F = E_q[ln q(s) - ln P(o | s) - ln P(s)]
        entropy = -np.sum(self.q_s * np.log(self.q_s + 1e-12))
        energy = -np.sum(self.q_s * (np.log(likelihood + 1e-12) + np.log(self.model.D + 1e-12)))
        vfe = energy + entropy

        return vfe

    def compute_expected_free_energy(self, action: int) -> float:
        """
        Compute Expected Free Energy (EFE) for a given action:

        G(π) = Risk + Ambiguity

        Risk = D_KL[ q(o | π) || P(o) ]
        Ambiguity = E_q(s) [ H[P(o | s)] ]

        Where:
        - Risk: Expected divergence from preferred observations
        - Ambiguity: Expected uncertainty in state-observation mapping
        """
        # Predicted next state distribution: q(s') = sum_s P(s' | s, a) q(s)
        q_next = self.model.B[:, :, action] @ self.q_s

        # Predicted observations: q(o) = sum_s P(o | s) q(s')
        q_obs = self.model.A @ q_next

        # Risk term: D_KL [ q(o) || P(o) ]
        # Use C as log preferences
        risk = np.sum(q_obs * (np.log(q_obs + 1e-12) - self.model.C))

        # Ambiguity term: E [ H[P(o | s)] ]
        # H = -sum_o P(o | s) ln P(o | s)
        entropy_A = -np.sum(self.model.A * np.log(self.model.A + 1e-12), axis=0)
        ambiguity = np.sum(q_next * entropy_A)

        # Total EFE = Risk + Ambiguity
        efe = risk + ambiguity

        return efe

    def plan_action(self) -> Tuple[int, np.ndarray]:
        """
        Select action that minimizes Expected Free Energy.

        Returns:
            (selected_action, EFE_values_for_all_actions)
        """
        efe_values = np.zeros(self.model.n_actions)

        for action in range(self.model.n_actions):
            efe_values[action] = self.compute_expected_free_energy(action)

        # Select action with minimal EFE
        best_action = int(np.argmin(efe_values))
        self.efe_history.append(efe_values.copy())

        return best_action, efe_values

    def step(self, observation: int) -> Tuple[int, float, np.ndarray]:
        """Complete perception-action cycle: observe -> infer -> act"""
        vfe = self.infer_states(observation)
        action, efe = self.plan_action()
        return action, vfe, efe


class MarkovDecisionProcess:
    """
    MDP wrapper for environment dynamics
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        initial_state: int = 0
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_matrix = transition_matrix  # [next, current, action]
        self.observation_matrix = observation_matrix  # [obs, state]
        self.current_state = initial_state

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Execute action and return (observation, reward, done, info)"""
        # Transition to next state
        transition_probs = self.transition_matrix[:, self.current_state, action]
        self.current_state = np.random.choice(self.n_states, p=transition_probs)

        # Generate observation
        obs_probs = self.observation_matrix[:, self.current_state]
        observation = np.random.choice(len(obs_probs), p=obs_probs)

        return observation, 0.0, False, {'true_state': self.current_state}

    def reset(self, state: Optional[int] = None) -> int:
        """Reset environment"""
        if state is not None:
            self.current_state = state
        return self.current_state


def run_active_inference_simulation(
    agent: ActiveInferenceAgentAdvanced,
    env: MarkovDecisionProcess,
    n_steps: int = 100
) -> Dict:
    """
    Run a complete active inference simulation.

    Returns simulation statistics.
    """
    observations = []
    actions = []
    vfe_values = []
    efe_values = []
    beliefs = []
    true_states = []

    for step in range(n_steps):
        # Get observation
        obs = np.random.choice(
            len(env.observation_matrix),
            p=env.observation_matrix[:, env.current_state]
        )

        # Active inference step
        action, vfe, efe = agent.step(obs)

        # Environment step
        env.step(action)

        # Record
        observations.append(obs)
        actions.append(action)
        vfe_values.append(vfe)
        efe_values.append(efe)
        beliefs.append(agent.q_s.copy())
        true_states.append(env.current_state)

    return {
        'observations': observations,
        'actions': actions,
        'vfe_values': vfe_values,
        'efe_values': efe_values,
        'beliefs': beliefs,
        'true_states': true_states,
        'final_belief': agent.q_s,
    }

"""
Stage 2: Scene Builder
Instantiates GenerativeModel from structured scene configuration
"""

import numpy as np
from typing import Dict, Any, Optional

from .generative_model import DiscreteGenerativeModel
from .language_parser import LanguageParser


class SceneBuilder:
    """
    Builds a complete generative model from scene configuration.
    Creates observation model, transition model, preferences, and initial state.
    """

    def __init__(self, parser: Optional[LanguageParser] = None):
        self.parser = parser or LanguageParser()

    def build_from_prompt(self, prompt: str) -> DiscreteGenerativeModel:
        """Build model directly from natural language prompt"""
        config = self.parser.parse(prompt)
        return self.build(config)

    def build(self, scene_config: Dict[str, Any]) -> DiscreteGenerativeModel:
        """
        Build a generative model from scene configuration.

        Args:
            scene_config: Configuration dictionary from LanguageParser

        Returns:
            Instantiated DiscreteGenerativeModel
        """
        agent_type = scene_config["agent_type"]
        agent_config = self.parser.get_agent_config(agent_type)

        # For this implementation:
        # State factors -> composite state space
        # We'll use a simplified state representation:
        #   position: 3 states (low food, near water, safe)
        #   hunger: 3 states (low, medium, high)
        #   health: 3 states (low, medium, high)
        #   age: 4 states (cub, juvenile, adult, elderly)

        n_state_factors = len(agent_config.get("state_factors", 4))

        # Composite state size: 3 (pos) x 3 (hunger) x 3 (health) x 4 (age) = 108
        # For simplicity, we'll use a reduced state space
        n_states = 27  # 3^3 composite states

        # Observations: noisy perception of state
        n_observations = 9

        # Actions: from agent config
        actions = agent_config.get("actions", ["move", "drink", "hunt", "eat", "rest", "mate", "nurture"])
        n_actions = len(actions)

        # Build observation model A
        A = self._build_observation_model(n_observations, n_states)

        # Build transition model B
        B = self._build_transition_model(n_states, n_actions)

        # Build preferences C
        C = self._build_preferences(n_observations, scene_config)

        # Build initial belief D
        D = self._build_initial_belief(n_states, scene_config)

        model = DiscreteGenerativeModel(
            n_states=n_states,
            n_observations=n_observations,
            n_actions=n_actions,
            observation_matrix=A,
            transition_tensor=B,
            preferences=C,
            initial_state_dist=D
        )

        # Store metadata
        model.scene_config = scene_config
        model.agent_config = agent_config

        return model

    def _build_observation_model(self, n_obs: int, n_states: int) -> np.ndarray:
        """Build observation likelihood matrix A[observation, state]"""
        # Mostly accurate observations with noise
        A = np.zeros((n_obs, n_states))

        # Each state maps to a primary observation
        for s in range(n_states):
            primary_obs = s % n_obs
            A[primary_obs, s] = 0.8
            # Add uniform noise
            A[:, s] += 0.2 / n_obs
            A[:, s] /= np.sum(A[:, s])

        return A

    def _build_transition_model(self, n_states: int, n_actions: int) -> np.ndarray:
        """Build transition tensor B[next_state, current_state, action]"""
        B = np.zeros((n_states, n_states, n_actions))

        for action in range(n_actions):
            for state in range(n_states):
                # Each action has some effect
                if action == 0:  # move
                    # Move to adjacent state
                    next_state = (state + 3) % n_states
                    B[next_state, state, action] = 0.7
                    # Stay in current state
                    B[state, state, action] = 0.3
                elif action == 1:  # drink
                    # Drinking improves hydration state
                    B[state, state, action] = 0.9
                    # Minor state change
                    B[(state + 1) % n_states, state, action] = 0.1
                else:
                    # Other actions have subtle state changes
                    B[state, state, action] = 0.8
                    for offset in [-2, -1, 1, 2]:
                        next_state = (state + offset) % n_states
                        B[next_state, state, action] = 0.05

                # Normalize
                B[:, state, action] /= np.sum(B[:, state, action])

        return B

    def _build_preferences(self, n_obs: int, config: Dict[str, Any]) -> np.ndarray:
        """Build preference vector C (log probabilities)"""
        agent_type = config["agent_type"]
        agent_config = self.parser.get_agent_config(agent_type)
        prefs = agent_config.get("default_preferences", {})

        # Preferences over observations
        C = np.ones(n_obs) / n_obs  # uniform baseline

        # Prefer observations corresponding to good health and low hunger
        # Observations 0-2: good states, 3-5: medium, 6-8: bad
        C[:3] *= prefs.get("high_health", 0.7)
        C[3:6] *= 0.3
        C[6:] *= 0.1

        # Normalize and convert to log probabilities
        C = C / np.sum(C)
        C = np.log(C + 1e-12)

        return C

    def _build_initial_belief(self, n_states: int, config: Dict[str, Any]) -> np.ndarray:
        """Build initial state distribution D"""
        life_stage = config.get("life_stage", "cub")

        D = np.ones(n_states) / n_states  # uniform

        # Bias initial state based on life stage
        if life_stage == "cub":
            # Cubs start young and healthy
            D[:9] *= 2.0  # First third = young/healthy states
        elif life_stage == "adult":
            D[9:18] *= 2.0
        elif life_stage == "elderly":
            D[18:] *= 2.0

        D = D / np.sum(D)

        return D

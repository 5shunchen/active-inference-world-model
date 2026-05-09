"""
Stage 3: Tiger Ecology Model
Full generative model for tiger lifecycle with dynamic preferences and ecological interactions.

State Factors (each 3 states):
  - POSITION: 0=near_prey, 1=near_water, 2=safe_territory
  - HUNGER: 0=full, 1=moderate, 2=starving
  - HEALTH: 0=injured, 1=healthy, 2=excellent
  - AGE: 0=cub, 1=juvenile, 2=adult, 3=elderly (4 states)

Actions (7): move, drink, hunt, eat, rest, mate, nurture

Total composite states: 3 x 3 x 3 x 4 = 108 states
Observations: 27 (noisy state perception)
"""

import numpy as np
from typing import Tuple, Dict, Any

from .generative_model import GenerativeModel
from .distributions import Categorical


class TigerEcologyModel(GenerativeModel):
    """
    Full ecology model for tiger lifecycle with active inference.
    Extends GenerativeModel with tiger-specific dynamics and preferences.
    """

    # State factor indices
    FACTORS = ["position", "hunger", "health", "age"]
    POSITION_IDX = 0
    HUNGER_IDX = 1
    HEALTH_IDX = 2
    AGE_IDX = 3

    # Position states
    POS_NEAR_PREY = 0
    POS_NEAR_WATER = 1
    POS_SAFE = 2

    # Hunger states
    HUNGER_FULL = 0
    HUNGER_MODERATE = 1
    HUNGER_STARVING = 2

    # Health states
    HEALTH_INJURED = 0
    HEALTH_HEALTHY = 1
    HEALTH_EXCELLENT = 2

    # Age states
    AGE_CUB = 0
    AGE_JUVENILE = 1
    AGE_ADULT = 2
    AGE_ELDERLY = 3

    # Actions
    ACTION_NAMES = ["move", "drink", "hunt", "eat", "rest", "mate", "nurture"]
    A_MOVE = 0
    A_DRINK = 1
    A_HUNT = 2
    A_EAT = 3
    A_REST = 4
    A_MATE = 5
    A_NURTURE = 6

    def __init__(self, scene_config: Dict[str, Any] = None):
        # Dimensions: 3x3x3x4 composite states
        self.factor_dims = [3, 3, 3, 4]
        n_states = int(np.prod(self.factor_dims))  # 108
        n_observations = 27  # Grouped observations
        n_actions = 7

        super().__init__(n_states, n_observations, n_actions)

        self.scene_config = scene_config or {}
        self._build_models()

    def _build_models(self):
        """Build all model components"""
        self._build_observation_model()
        self._build_transition_model()
        self._build_preferences()
        self._build_initial_belief()

    def _build_observation_model(self):
        """Build A matrix: P(observation | state)"""
        n_obs = self.n_observations
        n_states = self.n_states

        A = np.zeros((n_obs, n_states))

        for state in range(n_states):
            # Decode composite state
            pos, hunger, health, age = self._decode_state(state)

            # Primary observation based on dominant state factors
            obs_group = (pos * 9 + hunger * 3 + health) % n_obs

            # High probability of correct observation
            A[obs_group, state] = 0.85

            # Add noise to neighboring observations
            for offset in [-2, -1, 1, 2]:
                neighbor_obs = (obs_group + offset) % n_obs
                A[neighbor_obs, state] = 0.03

            # Normalize
            A[:, state] /= np.sum(A[:, state])

        self._A = A

    def _build_transition_model(self):
        """Build B tensor: P(next_state | state, action)"""
        n_states = self.n_states
        n_actions = self.n_actions

        B = np.zeros((n_states, n_states, n_actions))

        for state in range(n_states):
            pos, hunger, health, age = self._decode_state(state)

            for action in range(n_actions):
                # Start with base transition probabilities
                next_pos, next_hunger, next_health, next_age = pos, hunger, health, age

                # Apply action effects
                if action == self.A_MOVE:
                    # Move to a different position
                    next_pos = (pos + 1) % 3
                    # Moving burns calories
                    next_hunger = min(hunger + 1, 2)

                elif action == self.A_DRINK:
                    # Only effective at water
                    if pos == self.POS_NEAR_WATER:
                        pass  # Drinking doesn't directly change factors
                    else:
                        next_pos = self.POS_NEAR_WATER  # Must move to drink
                    next_hunger = max(hunger - 1, 0)  # Drinking reduces hunger slightly

                elif action == self.A_HUNT:
                    # Only effective at prey location
                    if pos == self.POS_NEAR_PREY:
                        # Hunting can lead to injury (especially for young/old)
                        injury_risk = 0.3 if age in [self.AGE_CUB, self.AGE_ELDERLY] else 0.1
                        if np.random.random() < injury_risk:
                            next_health = max(health - 1, 0)
                    else:
                        next_pos = self.POS_NEAR_PREY  # Move to hunting grounds
                    next_hunger = min(hunger + 1, 2)  # Hunting burns calories

                elif action == self.A_EAT:
                    # Only effective if just hunted or at prey
                    if pos == self.POS_NEAR_PREY:
                        next_hunger = max(hunger - 2, 0)  # Eating reduces hunger significantly
                        next_health = min(health + 1, 2)  # Eating improves health

                elif action == self.A_REST:
                    # Resting improves health and position is safe
                    if pos == self.POS_SAFE:
                        next_health = min(health + 1, 2)
                    next_hunger = min(hunger + 1, 2)  # Still burn calories while resting

                elif action == self.A_MATE:
                    # Only effective as adult, health cost
                    if age == self.AGE_ADULT and health >= self.HEALTH_HEALTHY:
                        next_health = max(health - 1, 0)
                    next_hunger = min(hunger + 1, 2)

                elif action == self.A_NURTURE:
                    # Nurturing offspring (for adults)
                    if age == self.AGE_ADULT:
                        next_hunger = min(hunger + 1, 2)

                # Aging: small probability of aging each step
                if age < self.AGE_ELDERLY and np.random.random() < 0.02:
                    next_age = age + 1

                # Encode next state
                next_state = self._encode_state(next_pos, next_hunger, next_health, next_age)

                # Main transition
                B[next_state, state, action] = 0.7

                # Add noise to neighboring states
                for offset in [-3, -1, 1, 3]:
                    noisy_next = (next_state + offset) % n_states
                    B[noisy_next, state, action] = 0.075

                # Normalize
                B[:, state, action] /= np.sum(B[:, state, action])

        self._B = B

    def _build_preferences(self):
        """Build C vector: preferences over observations"""
        n_obs = self.n_observations
        C = np.ones(n_obs)

        # Prefer low hunger, good health, safe position
        for state in range(self.n_states):
            pos, hunger, health, age = self._decode_state(state)
            obs_group = (pos * 9 + hunger * 3 + health) % n_obs

            # Base preference weight
            weight = 1.0

            # Prefer not being hungry
            if hunger == self.HUNGER_STARVING:
                weight *= 0.1
            elif hunger == self.HUNGER_FULL:
                weight *= 3.0

            # Prefer good health
            if health == self.HEALTH_INJURED:
                weight *= 0.2
            elif health == self.HEALTH_EXCELLENT:
                weight *= 2.0

            # Prefer safe position
            if pos == self.POS_SAFE:
                weight *= 1.5
            elif pos == self.POS_NEAR_WATER:
                weight *= 1.2

            C[obs_group] *= weight

        # Normalize and convert to log probabilities
        C = C / np.sum(C)
        self._C = np.log(C + 1e-12)

    def _build_initial_belief(self):
        """Build D vector: initial state distribution"""
        life_stage = self.scene_config.get("life_stage", "cub")

        D = np.ones(self.n_states)

        for state in range(self.n_states):
            pos, hunger, health, age = self._decode_state(state)

            # Bias based on life stage
            if life_stage == "cub" and age == self.AGE_CUB:
                D[state] *= 5.0
                D[state] *= 3.0 if health == self.HEALTH_HEALTHY else 1.0
            elif life_stage == "adult" and age == self.AGE_ADULT:
                D[state] *= 5.0
                D[state] *= 3.0 if health == self.HEALTH_EXCELLENT else 1.0
            elif life_stage == "elderly" and age == self.AGE_ELDERLY:
                D[state] *= 5.0

            # Start in safe territory
            if pos == self.POS_SAFE:
                D[state] *= 3.0

        D = D / np.sum(D)
        self._D = D

    def _decode_state(self, state_idx: int) -> Tuple[int, int, int, int]:
        """Decode composite state index into factor values"""
        remaining = state_idx
        factors = []
        for dim in reversed(self.factor_dims):
            factors.append(remaining % dim)
            remaining = remaining // dim
        return tuple(reversed(factors))  # (pos, hunger, health, age)

    def _encode_state(self, pos: int, hunger: int, health: int, age: int) -> int:
        """Encode factor values into composite state index"""
        state = 0
        multiplier = 1
        for factor, dim in zip([pos, hunger, health, age], self.factor_dims):
            state = state * dim + factor
        return state

    def get_observation_model(self):
        from .distributions import LikelihoodModel
        return LikelihoodModel(self._A)

    def get_transition_model(self):
        from .distributions import TransitionModel
        return TransitionModel(self._B)

    def _sample_transition(self, state, action):
        probs = self._B[:, state, action]
        return Categorical(probs).sample()[0]

    def get_preferences(self, state=None):
        return self._C

    def get_initial_belief(self):
        return Categorical(self._D)

    def step_dynamics(self, state: int, action: int) -> int:
        """True environment dynamics"""
        return self._sample_transition(state, action)

    def analyze_state(self, state_idx: int) -> Dict[str, str]:
        """Get human-readable state description"""
        pos, hunger, health, age = self._decode_state(state_idx)

        pos_names = ["near_prey", "near_water", "safe_territory"]
        hunger_names = ["full", "moderate", "starving"]
        health_names = ["injured", "healthy", "excellent"]
        age_names = ["cub", "juvenile", "adult", "elderly"]

        return {
            "position": pos_names[pos],
            "hunger": hunger_names[hunger],
            "health": health_names[health],
            "age": age_names[age]
        }

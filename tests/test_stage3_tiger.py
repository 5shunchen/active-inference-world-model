"""
Tests for Stage 3: Tiger Ecology Model
Validates tiger can survive, grow, and complete lifecycle
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tiger_ecology_model import TigerEcologyModel
from src.world_simulator import WorldSimulator
from src.inference_engine import ActiveInferenceAgent


class TestTigerEcologyModel:
    def test_model_initializes(self):
        model = TigerEcologyModel()
        assert model.n_states == 108  # 3*3*3*4
        assert model.n_observations == 27
        assert model.n_actions == 7

    def test_state_encoding_decoding(self):
        model = TigerEcologyModel()

        # Test encode then decode gives same values
        pos, hunger, health, age = 1, 2, 1, 2
        state = model._encode_state(pos, hunger, health, age)
        decoded = model._decode_state(state)

        assert decoded == (pos, hunger, health, age)

    def test_observation_model_valid(self):
        model = TigerEcologyModel()

        obs_model = model.get_observation_model()
        for s in range(model.n_states):
            probs = obs_model.matrix[:, s]
            assert np.isclose(np.sum(probs), 1.0, atol=0.01)

    def test_initial_belief_biased_by_life_stage(self):
        config = {"life_stage": "cub"}
        model = TigerEcologyModel(config)

        belief = model.get_initial_belief().probs

        # Count probability mass for cub vs adult states
        cub_prob = 0
        adult_prob = 0

        for s in range(model.n_states):
            pos, hunger, health, age = model._decode_state(s)
            if age == TigerEcologyModel.AGE_CUB:
                cub_prob += belief[s]
            elif age == TigerEcologyModel.AGE_ADULT:
                adult_prob += belief[s]

        # Most probability should be on cub states
        assert cub_prob > adult_prob * 3

    def test_analyze_state_returns_description(self):
        model = TigerEcologyModel()
        description = model.analyze_state(0)

        assert "position" in description
        assert "hunger" in description
        assert "health" in description
        assert "age" in description


class TestTigerLifecycle:
    def test_tiger_survives_multiple_steps(self):
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        for _ in range(50):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

        # Should have run without errors
        assert len(world.trajectory.states) > 1

    def test_tiger_lifecycle_stages_present(self):
        """Test that simulation visits different age states"""
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        age_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for _ in range(200):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

            pos, hunger, health, age = model._decode_state(world.get_true_state())
            age_counts[age] += 1

        # Should see different age groups
        ages_seen = sum(1 for count in age_counts.values() if count > 0)
        assert ages_seen >= 2

    def test_tiger_actions_vary(self):
        """Test tiger takes different actions"""
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        actions_taken = set()
        for _ in range(100):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            actions_taken.add(action)
            world.step(action)

        # Should take more than one type of action (stochastic behavior)
        assert len(actions_taken) >= 2

    def test_free_energy_stabilizes(self):
        """Test free energy doesn't diverge"""
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        fe_values = []
        for _ in range(50):
            obs = world.observe()
            fe = agent.perceive(obs)
            fe_values.append(fe)
            action = agent.act()
            world.step(action)

        # Free energy should be finite
        assert np.mean(fe_values) < 10
        assert not np.any(np.isnan(fe_values))
        assert not np.any(np.isinf(fe_values))

    def test_tiger_maintains_health(self):
        """Test tiger doesn't stay injured continuously"""
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        health_values = []
        for _ in range(100):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

            pos, hunger, health, age = model._decode_state(world.get_true_state())
            health_values.append(health)

        # Average health should not be at minimum
        avg_health = np.mean(health_values)
        assert avg_health > 0.3  # Should maintain some health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

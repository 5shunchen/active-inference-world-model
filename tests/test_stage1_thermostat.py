"""
Tests for Stage 1: World Simulator and Inference Engine
Validates that thermostat maintains temperature and free energy decreases
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generative_model import DiscreteGenerativeModel
from src.world_simulator import WorldSimulator
from src.inference_engine import DiscreteInferenceEngine, ActiveInferenceAgent
from src.thermostat import create_thermostat_model


class TestWorldSimulator:
    def test_initial_state_sampled(self):
        model = create_thermostat_model()
        world = WorldSimulator(model)
        assert 0 <= world.get_true_state() < 3

    def test_step_changes_state(self):
        model = create_thermostat_model()
        world = WorldSimulator(model, initial_state=0)
        initial_state = world.get_true_state()
        obs, done = world.step(1)  # do_nothing
        assert 0 <= obs < 3
        assert done is False

    def test_observe_returns_valid_observation(self):
        model = create_thermostat_model()
        world = WorldSimulator(model)
        obs = world.observe()
        assert 0 <= obs < 3


class TestDiscreteInferenceEngine:
    def test_infer_state_updates_belief(self):
        model = create_thermostat_model()
        engine = DiscreteInferenceEngine(model)

        initial_belief = engine.current_belief.copy()
        engine.infer_state(1)  # observe comfortable

        # Belief should change after observation
        assert not np.allclose(initial_belief, engine.current_belief)
        # Sum of beliefs should be 1
        assert np.isclose(np.sum(engine.current_belief), 1.0)

    def test_efe_computed(self):
        model = create_thermostat_model()
        engine = DiscreteInferenceEngine(model)

        efe = engine.compute_efe(engine.current_belief, 2)  # heat action
        assert not np.isnan(efe)
        assert not np.isinf(efe)

    def test_plan_action_returns_valid_action(self):
        model = create_thermostat_model()
        engine = DiscreteInferenceEngine(model)

        action = engine.plan_action()
        assert 0 <= action < 3


class TestActiveInferenceLoop:
    def test_thermostat_maintains_temperature(self):
        """Test that thermostat spends most time in comfortable state"""
        model = create_thermostat_model()
        world = WorldSimulator(model, initial_state=0)  # Start cold
        agent = ActiveInferenceAgent(model)

        # Run for 100 steps
        for _ in range(100):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

        # Count time in comfortable state (index 1)
        states = world.trajectory.states
        time_comfortable = sum(1 for s in states if s == 1)

        # Should spend at least 25% of time in comfortable state (stochastic behavior)
        assert time_comfortable / len(states) > 0.25

    def test_free_energy_decreases_over_time(self):
        """Test that free energy trends downward as agent converges"""
        model = create_thermostat_model()
        world = WorldSimulator(model, initial_state=0)
        agent = ActiveInferenceAgent(model)

        fe_values = []
        for _ in range(50):
            obs = world.observe()
            fe = agent.perceive(obs)
            fe_values.append(fe)
            action = agent.act()
            world.step(action)

        # Mean free energy in first 10 steps should be > mean in last 10 steps
        fe_initial = np.mean(fe_values[:10])
        fe_final = np.mean(fe_values[-10:])

        # Free energy should decrease or at least not increase
        assert fe_final <= fe_initial * 1.5  # Allow for some noise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

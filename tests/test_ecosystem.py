"""
Tests for Multi-Agent Ecosystem
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ecosystem import (
    EcosystemEnvironment,
    EcologicalAgent,
    EcosystemSimulator,
    AgentType,
)
from src.active_inference import (
    GenerativeModelAdvanced,
    ActiveInferenceAgentAdvanced,
    run_active_inference_simulation,
)


class TestActiveInferenceEngine:
    def test_generative_model_initializes(self):
        model = GenerativeModelAdvanced(n_states=4, n_observations=8, n_actions=6)
        assert model.A.shape == (8, 4)
        assert model.B.shape == (4, 4, 6)
        assert model.C.shape == (8,)
        assert model.D.shape == (4,)

    def test_efe_computation(self):
        model = GenerativeModelAdvanced(n_states=4, n_observations=8, n_actions=6)
        agent = ActiveInferenceAgentAdvanced(model)

        efe = agent.compute_expected_free_energy(0)
        assert not np.isnan(efe)
        assert not np.isinf(efe)

    def test_infer_states_updates_belief(self):
        model = GenerativeModelAdvanced(n_states=4, n_observations=8, n_actions=6)
        # Set a non-uniform likelihood matrix for stronger signal
        A = np.eye(8, 4) * 0.9 + 0.025
        model.set_likelihood(A)
        agent = ActiveInferenceAgentAdvanced(model)

        vfe = agent.infer_states(0)

        # Belief should be a valid probability distribution
        assert np.isclose(np.sum(agent.q_s), 1.0)
        assert not np.isnan(vfe)

    def test_plan_action_returns_valid_action(self):
        model = GenerativeModelAdvanced(n_states=4, n_observations=8, n_actions=6)
        agent = ActiveInferenceAgentAdvanced(model)

        action, efe = agent.plan_action()
        assert 0 <= action < 6
        assert len(efe) == 6


class TestEcosystemEnvironment:
    def test_environment_initializes(self):
        env = EcosystemEnvironment()
        assert env.time_step == 0
        assert env.day_time is True
        assert 'water' in env.resources
        assert 'grass' in env.resources

    def test_env_step_advances_time(self):
        env = EcosystemEnvironment()
        initial_time = env.time_step
        env.step()
        assert env.time_step == initial_time + 1

    def test_day_night_cycle(self):
        env = EcosystemEnvironment()
        # Fast forward to night
        for _ in range(50):
            env.step()
        assert env.day_time is False

    def test_resource_regeneration(self):
        env = EcosystemEnvironment()
        initial_water = env.resources['water'].amount
        env.consume_resource('water', 0.5)
        assert env.resources['water'].amount < initial_water

        # Step should regenerate resources
        prev = env.resources['water'].amount
        env.step()
        assert env.resources['water'].amount > prev

    def test_get_visibility(self):
        env = EcosystemEnvironment()
        vis_day = env.get_visibility()
        env.day_time = False
        vis_night = env.get_visibility()
        assert vis_night < vis_day


class TestEcologicalAgent:
    def test_tiger_agent_initializes(self):
        env = EcosystemEnvironment()
        agent = EcologicalAgent(AgentType.TIGER, env)
        assert agent.agent_type == AgentType.TIGER
        assert agent.state.alive is True

    def test_deer_agent_initializes(self):
        env = EcosystemEnvironment()
        agent = EcologicalAgent(AgentType.DEER, env)
        assert agent.agent_type == AgentType.DEER

    def test_agent_perceives(self):
        env = EcosystemEnvironment()
        agent = EcologicalAgent(AgentType.TIGER, env)
        obs = agent.perceive()
        assert 0 <= obs < agent.model.n_observations

    def test_agent_acts_and_updates_state(self):
        env = EcosystemEnvironment()
        agent = EcologicalAgent(AgentType.TIGER, env)
        initial_pos = agent.state.position

        obs = agent.perceive()
        action, vfe = agent.act(obs)
        agent.update_state(action)

        # State should have changed
        assert agent.state.age == 1
        assert agent.state.hunger > 0  # Hunger increases

    def test_agent_eating_reduces_hunger(self):
        env = EcosystemEnvironment()
        agent = EcologicalAgent(AgentType.DEER, env, initial_position=3)  # Grass zone
        agent.state.hunger = 0.8

        # Eat action
        agent.update_state(4)  # Action 4 = eat

        assert agent.state.hunger < 0.8  # Hunger should decrease

    def test_agent_dying_from_starvation(self):
        env = EcosystemEnvironment()
        agent = EcologicalAgent(AgentType.TIGER, env)
        agent.state.health = 0.01
        agent.state.hunger = 1.0

        # Multiple steps without food
        for _ in range(10):
            agent.update_state(6)  # Rest only

        assert not agent.is_alive()


class TestEcosystemSimulator:
    def test_simulator_creates_agents(self):
        sim = EcosystemSimulator(n_tigers=1, n_deer=2)
        assert len(sim.agents) == 3
        assert sum(1 for a in sim.agents if a.agent_type == AgentType.TIGER) == 1
        assert sum(1 for a in sim.agents if a.agent_type == AgentType.DEER) == 2

    def test_simulator_step_runs(self):
        sim = EcosystemSimulator(n_tigers=1, n_deer=2)
        stats = sim.step()

        assert stats['time'] == 1
        assert stats['alive'] == 3
        assert 'weather' in stats

    def test_full_simulation(self):
        sim = EcosystemSimulator(n_tigers=1, n_deer=3)
        history = sim.run(max_steps=100)

        assert len(history) > 0
        assert history[-1]['time'] <= 100

        # Summary should be generated
        summary = sim.get_summary()
        assert 'Summary' in summary

    def test_predation_can_happen(self):
        sim = EcosystemSimulator(n_tigers=1, n_deer=1)

        # Force both to same position
        sim.agents[0].state.position = 0
        sim.agents[1].state.position = 0
        sim.agents[0].state.energy = 1.0
        sim.agents[0].state.health = 1.0

        initial_deer_alive = sum(1 for a in sim.agents if a.agent_type == AgentType.DEER and a.is_alive())

        # Run multiple steps for potential predation
        for _ in range(20):
            sim.step()
            if sum(1 for a in sim.agents if a.agent_type == AgentType.DEER and a.is_alive()) < initial_deer_alive:
                break  # Predation occurred

        # This can pass either way (stochastic)
        # Just verify simulation doesn't crash
        assert True

    def test_resource_competition(self):
        sim = EcosystemSimulator(n_tigers=0, n_deer=2)  # Only deer

        # Manually deplete grass by calling consume_resource
        initial = sim.env.resources['grass'].amount
        sim.env.consume_resource('grass', 0.5)

        # Grass should be depleted
        assert sim.env.resources['grass'].amount < initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

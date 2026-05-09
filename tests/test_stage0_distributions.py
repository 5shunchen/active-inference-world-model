"""
Tests for Stage 0: Distributions and Generative Model
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.distributions import Categorical, MultivariateGaussian, LikelihoodModel, TransitionModel
from src.generative_model import DiscreteGenerativeModel


class TestCategorical:
    def test_sample_returns_valid_samples(self):
        cat = Categorical([0.2, 0.3, 0.5])
        samples = cat.sample(100)
        assert len(samples) == 100
        assert all(0 <= s < 3 for s in samples)

    def test_log_prob_returns_log_probabilities(self):
        cat = Categorical([0.2, 0.3, 0.5])
        lp0 = cat.log_prob(0)
        lp1 = cat.log_prob(1)
        lp2 = cat.log_prob(2)
        assert np.isclose(np.exp(lp0), 0.2, atol=0.01)
        assert np.isclose(np.exp(lp1), 0.3, atol=0.01)
        assert np.isclose(np.exp(lp2), 0.5, atol=0.01)

    def test_normalizes_probabilities(self):
        cat = Categorical([1, 1, 1])
        assert np.allclose(cat.probs, [1/3, 1/3, 1/3])


class TestMultivariateGaussian:
    def test_sample_returns_correct_shape(self):
        gauss = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        samples = gauss.sample(10)
        assert samples.shape == (10, 2)

    def test_log_prob_returns_values(self):
        gauss = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        lp = gauss.log_prob([0, 0])
        assert not np.isnan(lp) and not np.isinf(lp)


class TestLikelihoodModel:
    def test_sample_observation_returns_valid_observation(self):
        A = np.zeros((3, 2))
        A[:, 0] = [0.8, 0.1, 0.1]
        A[:, 1] = [0.1, 0.1, 0.8]
        model = LikelihoodModel(A)
        obs = model.sample_observation(0)
        assert 0 <= obs < 3

    def test_get_observation_prob(self):
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        model = LikelihoodModel(A)
        probs = model.get_observation_prob(0)
        assert np.allclose(probs, [0.9, 0.1])


class TestTransitionModel:
    def test_sample_next_state(self):
        B = np.zeros((2, 2, 1))
        B[:, 0, 0] = [0.7, 0.3]
        B[:, 1, 0] = [0.4, 0.6]
        model = TransitionModel(B)
        next_state = model.sample_next_state(0, 0)
        assert 0 <= next_state < 2


class TestDiscreteGenerativeModel:
    def test_creates_valid_model(self):
        n_states = 2
        n_obs = 2
        n_actions = 1
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        B = np.zeros((2, 2, 1))
        B[:, 0, 0] = [0.7, 0.3]
        B[:, 1, 0] = [0.4, 0.6]
        C = np.log(np.array([0.5, 0.5]))
        D = np.array([0.5, 0.5])
        model = DiscreteGenerativeModel(n_states, n_obs, n_actions, A, B, C, D)
        assert model.n_states == n_states
        assert model.n_observations == n_obs
        assert model.n_actions == n_actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

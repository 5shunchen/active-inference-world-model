"""
Tests for Stage 2: Language Parser and Scene Builder
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.language_parser import LanguageParser
from src.scene_builder import SceneBuilder


class TestLanguageParser:
    def test_parser_initializes_with_templates(self):
        parser = LanguageParser()
        assert "prompt_templates" in parser.templates
        assert "agents" in parser.templates
        assert "biomes" in parser.templates

    def test_parse_tiger_life_story_exact_match(self):
        parser = LanguageParser()
        config = parser.parse("老虎的生命历程")

        assert config["biome"] == "jungle"
        assert config["agent_type"] == "tiger"
        assert config["life_stage"] == "cub"
        assert "prey" in config["entities"]
        assert "water_source" in config["entities"]

    def test_parse_tiger_english(self):
        parser = LanguageParser()
        config = parser.parse("tiger life story")

        assert config["agent_type"] == "tiger"
        assert config["biome"] == "jungle"

    def test_keyword_based_parsing(self):
        parser = LanguageParser()
        config = parser.parse("我想看一只成年老虎在草原上的生活")

        assert config["agent_type"] == "tiger"
        assert config["biome"] == "savanna"
        assert config["life_stage"] == "adult"

    def test_get_agent_config(self):
        parser = LanguageParser()
        tiger_config = parser.get_agent_config("tiger")

        assert "state_factors" in tiger_config
        assert "actions" in tiger_config
        assert "hunt" in tiger_config["actions"]
        assert "eat" in tiger_config["actions"]

    def test_get_biome_config(self):
        parser = LanguageParser()
        jungle_config = parser.get_biome_config("jungle")

        assert "terrain_features" in jungle_config
        assert "trees" in jungle_config["terrain_features"]


class TestSceneBuilder:
    def test_build_from_prompt(self):
        builder = SceneBuilder()
        model = builder.build_from_prompt("老虎的生命历程")

        assert model.n_states > 0
        assert model.n_observations > 0
        assert model.n_actions > 0

    def test_model_has_metadata(self):
        builder = SceneBuilder()
        model = builder.build_from_prompt("老虎的生命历程")

        assert hasattr(model, "scene_config")
        assert hasattr(model, "agent_config")
        assert model.scene_config["agent_type"] == "tiger"

    def test_observation_matrix_is_valid(self):
        builder = SceneBuilder()
        model = builder.build_from_prompt("老虎的生命历程")

        A = model.get_observation_model().matrix

        # Check shape
        assert A.shape == (model.n_observations, model.n_states)

        # Check columns sum to 1 (probability distributions)
        for s in range(model.n_states):
            assert np.isclose(np.sum(A[:, s]), 1.0, atol=0.01)

    def test_transition_tensor_is_valid(self):
        builder = SceneBuilder()
        model = builder.build_from_prompt("老虎的生命历程")

        B = model.get_transition_model().tensor

        # Check shape
        assert B.shape == (model.n_states, model.n_states, model.n_actions)

        # Check each column sums to 1
        for s in range(model.n_states):
            for a in range(model.n_actions):
                assert np.isclose(np.sum(B[:, s, a]), 1.0, atol=0.01)

    def test_preferences_are_valid_log_probs(self):
        builder = SceneBuilder()
        model = builder.build_from_prompt("老虎的生命历程")

        C = model.get_preferences()

        assert len(C) == model.n_observations
        # Log probabilities should be negative (since probs < 1)
        assert all(C <= 0)

    def test_initial_belief_is_valid_distribution(self):
        builder = SceneBuilder()
        model = builder.build_from_prompt("老虎的生命历程")

        D = model.get_initial_belief().probs

        assert len(D) == model.n_states
        assert np.isclose(np.sum(D), 1.0, atol=0.01)
        assert all(D >= 0)

    def test_different_prompts_create_different_configs(self):
        builder = SceneBuilder()

        model1 = builder.build_from_prompt("老虎的生命历程")
        model2 = builder.build_from_prompt("鹿的一生")

        # Different agent types
        assert model1.scene_config["agent_type"] != model2.scene_config["agent_type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

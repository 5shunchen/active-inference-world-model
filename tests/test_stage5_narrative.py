"""
Tests for Stage 5: Narrative Generator
"""

import pytest
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tiger_ecology_model import TigerEcologyModel
from src.narrative_generator import NarrativeGenerator
from src.world_simulator import WorldSimulator
from src.inference_engine import ActiveInferenceAgent


class TestNarrativeGenerator:
    def test_generator_initializes(self):
        model = TigerEcologyModel()
        generator = NarrativeGenerator(model)
        assert generator.templates is not None
        assert 'birth' in generator.templates

    def test_detect_events_returns_events(self):
        model = TigerEcologyModel()
        generator = NarrativeGenerator(model)

        # Create simple trajectory visiting different positions
        states = []
        for pos in [0, 1, 2]:
            state = model._encode_state(pos, 1, 1, 1)
            states.extend([state] * 5)

        actions = [0] * len(states)  # move action

        events = generator.detect_events(states, actions)

        assert len(events) > 0
        assert events[0]['type'] == 'birth'  # First step is always birth

    def test_generate_narrative_produces_text(self):
        model = TigerEcologyModel()
        generator = NarrativeGenerator(model)

        states = [model._encode_state(0, 1, 1, 1)] * 10
        actions = [0] * 10

        events = generator.detect_events(states, actions)
        narrative = generator.generate_narrative(events)

        assert len(narrative) == len(events)
        assert all('time' in entry for entry in narrative)
        assert all('text' in entry for entry in narrative)
        assert all(len(entry['text']) > 0 for entry in narrative)

    def test_generate_summary(self):
        model = TigerEcologyModel()
        generator = NarrativeGenerator(model)

        states = [model._encode_state(0, 1, 1, 1)] * 50
        summary = generator.generate_summary(states)

        assert 'Life Summary' in summary
        assert 'Duration' in summary
        assert 'Final age' in summary

    def test_generate_subtitles_json(self):
        model = TigerEcologyModel()
        generator = NarrativeGenerator(model)

        narrative = [
            {'time': 0, 'text': 'A tiger is born', 'event_type': 'birth'},
            {'time': 10, 'text': 'The tiger rests', 'event_type': 'resting'},
        ]

        subtitles_json = generator.generate_subtitles_json(narrative)
        subtitles = json.loads(subtitles_json)

        assert len(subtitles) == 2
        assert 'start_time' in subtitles[0]
        assert 'end_time' in subtitles[0]
        assert 'text' in subtitles[0]

    def test_all_lifecycle_events_present(self):
        """Test narrative covers all key life stages"""
        model = TigerEcologyModel()
        generator = NarrativeGenerator(model)

        # Create a trajectory that goes through different ages
        states = []
        for age in [0, 1, 2, 3]:
            for pos in [0, 1, 2]:
                state = model._encode_state(pos, 1, 1, age)
                states.extend([state] * 5)

        actions = [0] * len(states)
        events = generator.detect_events(states, actions)
        narrative = generator.generate_narrative(events)

        # Should have birth and aging events
        event_types = set(e['type'] for e in events)
        assert 'birth' in event_types
        assert 'aging' in event_types

        # Narrative text should be present
        texts = [n['text'] for n in narrative]
        assert any('born' in t.lower() for t in texts) or any('birth' in t.lower() for t in texts)


class TestNarrativeIntegration:
    def test_full_simulation_narrative(self):
        """Test narrative generation from actual simulation"""
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        # Run simulation
        for _ in range(30):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

        # Generate narrative
        generator = NarrativeGenerator(model)
        result = generator.process_trajectory(world.trajectory)

        assert 'events' in result
        assert 'narrative' in result
        assert 'summary' in result
        assert 'subtitles_json' in result

        assert len(result['narrative']) > 0
        assert len(result['summary']) > 0

    def test_narrative_covers_birth_and_other_events(self):
        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        for _ in range(20):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

        generator = NarrativeGenerator(model)
        result = generator.process_trajectory(world.trajectory)

        # Birth should be in events
        event_types = [e['type'] for e in result['events']]
        assert 'birth' in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for Stage 4: Visual Renderer
"""

import pytest
import numpy as np
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tiger_ecology_model import TigerEcologyModel
from src.visual_renderer import VisualRenderer


class TestVisualRenderer:
    def test_renderer_initializes(self):
        model = TigerEcologyModel()
        renderer = VisualRenderer(model)
        assert renderer.width == 640
        assert renderer.height == 480

    def test_render_frame_returns_valid_array(self):
        model = TigerEcologyModel()
        renderer = VisualRenderer(model, width=320, height=240)

        state = model._encode_state(0, 1, 1, 1)  # Some valid state
        frame = renderer.render_frame(state, step=0)

        # Should return a numpy array with correct dimensions
        assert isinstance(frame, np.ndarray)
        assert frame.shape[0] == 240  # height
        assert frame.shape[1] == 320  # width

    def test_render_frame_for_different_states(self):
        model = TigerEcologyModel()
        renderer = VisualRenderer(model, width=160, height=120)

        # Test different positions
        positions = []
        for pos in [0, 1, 2]:
            state = model._encode_state(pos, 1, 1, 1)
            frame = renderer.render_frame(state, step=0)
            positions.append(frame)

        # Different positions should produce different frames
        # (different background colors)
        assert not np.allclose(positions[0], positions[1])

    def test_render_trajectory_returns_multiple_frames(self):
        model = TigerEcologyModel()
        renderer = VisualRenderer(model, width=160, height=120)

        states = [
            model._encode_state(0, 1, 1, 1),
            model._encode_state(1, 1, 1, 1),
            model._encode_state(2, 1, 1, 1),
        ]

        frames = renderer.render_trajectory(states)
        assert len(frames) == 3
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_save_video_creates_output(self):
        model = TigerEcologyModel()
        renderer = VisualRenderer(model, width=160, height=120)

        # Create some frames
        states = [model._encode_state(0, 1, 1, 1)] * 5
        frames = renderer.render_trajectory(states)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_video.mp4")
            result = renderer.save_video(frames, output_path)

            # Should have created something (either file or directory)
            assert os.path.exists(result)


class TestRendererIntegration:
    def test_render_from_simulation(self):
        """Test rendering frames from actual simulation"""
        from src.world_simulator import WorldSimulator
        from src.inference_engine import ActiveInferenceAgent

        model = TigerEcologyModel()
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        # Run short simulation
        for _ in range(5):
            obs = world.observe()
            agent.perceive(obs)
            action = agent.act()
            world.step(action)

        # Render frames
        renderer = VisualRenderer(model, width=160, height=120)
        frames = renderer.render_trajectory(world.trajectory.states)

        assert len(frames) == len(world.trajectory.states)
        assert all(f.shape == (120, 160, 3) or f.shape == (120, 160, 4) for f in frames)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

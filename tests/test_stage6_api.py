"""
Tests for Stage 6: End-to-End API Integration
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.life_story_api import LifestoryAPI, create_requirements_file


class TestLifestoryAPI:
    def test_api_generates_valid_result(self):
        """Test the full API flow without video rendering"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = LifestoryAPI.generate(
                prompt="给我一个老虎的生命历程",
                output_video_path=os.path.join(tmpdir, "test.mp4"),
                max_steps=50,
                render_video=False
            )

            # Check result structure
            assert 'prompt' in result
            assert 'video_path' in result
            assert 'narrative' in result
            assert 'summary' in result
            assert 'subtitles_json' in result
            assert 'metadata' in result

    def test_api_returns_valid_narrative(self):
        """Test API returns proper narrative content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = LifestoryAPI.generate(
                prompt="tiger life story",
                output_video_path=os.path.join(tmpdir, "test.mp4"),
                max_steps=30,
                render_video=False
            )

            assert len(result['narrative']) > 0
            assert 'Life Summary' in result['summary']
            assert len(result['summary']) > 0

    def test_api_metadata_complete(self):
        """Test metadata has all expected fields"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = LifestoryAPI.generate(
                prompt="tiger",
                output_video_path=os.path.join(tmpdir, "test.mp4"),
                max_steps=25,
                render_video=False
            )

            metadata = result['metadata']
            assert 'simulation_steps' in metadata
            assert 'actions_taken' in metadata
            assert 'mean_free_energy' in metadata
            assert 'agent_type' in metadata
            assert 'runtime_seconds' in metadata

            assert metadata['agent_type'] == 'tiger'
            assert metadata['simulation_steps'] > 0

    def test_api_chinese_prompt(self):
        """Test API works with Chinese prompts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = LifestoryAPI.generate(
                prompt="给我一个老虎的生命历程",
                output_video_path=os.path.join(tmpdir, "test.mp4"),
                max_steps=20,
                render_video=False
            )

            assert result['prompt'] == "给我一个老虎的生命历程"
            assert len(result['narrative']) > 0

    def test_subtitles_json_is_valid_json(self):
        """Test subtitles output is valid JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = LifestoryAPI.generate(
                prompt="tiger",
                output_video_path=os.path.join(tmpdir, "test.mp4"),
                max_steps=20,
                render_video=False
            )

            subtitles = json.loads(result['subtitles_json'])
            assert isinstance(subtitles, list)
            assert len(subtitles) > 0
            assert 'start_time' in subtitles[0]
            assert 'end_time' in subtitles[0]
            assert 'text' in subtitles[0]


class TestRequirements:
    def test_requirements_file_creation(self):
        """Test requirements file can be created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = os.path.join(tmpdir, "requirements.txt")
            create_requirements_file(req_path)

            assert os.path.exists(req_path)

            with open(req_path, 'r') as f:
                content = f.read()

            assert 'numpy' in content
            assert 'matplotlib' in content
            assert 'pytest' in content
            assert 'opencv-python' in content


class TestFullPipeline:
    def test_full_pipeline_with_video_rendering(self):
        """Test complete pipeline including video rendering"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "tiger_life.mp4")

            result = LifestoryAPI.generate(
                prompt="给我一个老虎的生命历程",
                output_video_path=video_path,
                max_steps=30,
                render_video=True
            )

            # Video should be created
            assert os.path.exists(result['video_path']) or os.path.isdir(result['video_path'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

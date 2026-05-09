"""
Stage 6: End-to-End Life Story API

Main entry point: LifestoryAPI.generate(prompt, output_video_path) -> dict

This is the fully integrated system that:
1. Parses natural language prompt
2. Builds generative model
3. Runs active inference simulation
4. Renders video
5. Generates narrative
6. Returns complete output
"""

import os
import json
import time
from typing import Dict, Any, Optional

from .language_parser import LanguageParser
from .scene_builder import SceneBuilder
from .tiger_ecology_model import TigerEcologyModel
from .world_simulator import WorldSimulator
from .inference_engine import ActiveInferenceAgent
from .visual_renderer import VisualRenderer
from .narrative_generator import NarrativeGenerator


class LifestoryAPI:
    """End-to-end API for generating life story videos"""

    @staticmethod
    def generate(
        prompt: str,
        output_video_path: str,
        max_steps: int = 500,
        render_video: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a life story video from a natural language prompt.

        Args:
            prompt: Natural language description (e.g., "给我一个老虎的生命历程")
            output_video_path: Path to save the MP4 video
            max_steps: Maximum simulation steps
            render_video: Whether to render video (False for testing)

        Returns:
            Dictionary with video path, narrative text, and metadata
        """
        start_time = time.time()

        # Step 1: Parse prompt and build scene config
        print(f"[API] Parsing prompt: {prompt}")
        parser = LanguageParser()
        scene_config = parser.parse(prompt)

        # Step 2: Build generative model for the agent
        print(f"[API] Building ecology model for: {scene_config['agent_type']}")
        model = TigerEcologyModel(scene_config)

        # Step 3: Run active inference simulation
        print(f"[API] Running simulation (max {max_steps} steps)...")
        world = WorldSimulator(model)
        agent = ActiveInferenceAgent(model)

        free_energy_values = []

        for step in range(max_steps):
            obs = world.observe()
            fe = agent.perceive(obs)
            free_energy_values.append(fe)
            action = agent.act()
            world.step(action)

        print(f"[API] Simulation completed: {len(world.trajectory.states)} states, "
              f"{len(world.trajectory.actions)} actions")

        # Step 4: Generate narrative
        print(f"[API] Generating narrative...")
        narrative_gen = NarrativeGenerator(model)
        narrative_result = narrative_gen.process_trajectory(world.trajectory)

        # Step 5: Render video (if enabled)
        video_path = None
        if render_video:
            print(f"[API] Rendering video...")
            renderer = VisualRenderer(model)
            video_path = renderer.generate_video_from_trajectory(
                world.trajectory, output_video_path
            )

        # Step 6: Prepare and return results
        total_time = time.time() - start_time

        result = {
            'prompt': prompt,
            'video_path': video_path,
            'narrative': narrative_result['narrative'],
            'summary': narrative_result['summary'],
            'subtitles_json': narrative_result['subtitles_json'],
            'metadata': {
                'simulation_steps': len(world.trajectory.states),
                'actions_taken': len(world.trajectory.actions),
                'mean_free_energy': sum(free_energy_values) / len(free_energy_values) if free_energy_values else 0,
                'agent_type': scene_config['agent_type'],
                'runtime_seconds': round(total_time, 2),
            }
        }

        print(f"[API] Generation complete in {total_time:.1f} seconds!")
        return result


def create_requirements_file(output_path: str = "requirements.txt"):
    """Create requirements.txt file for the project"""
    requirements = [
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pytest>=7.4.0",
        "opencv-python>=4.8.0",
    ]

    with open(output_path, 'w') as f:
        f.write('\n'.join(requirements) + '\n')

    print(f"[Setup] Created {output_path}")


if __name__ == "__main__":
    # Quick demo
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    result = LifestoryAPI.generate(
        prompt="给我一个老虎的生命历程",
        output_video_path=os.path.join(output_dir, "tiger_life.mp4"),
        max_steps=200,
        render_video=True
    )

    print("\n=== Summary ===")
    print(result['summary'])
    print(f"\nVideo saved to: {result['video_path']}")
    print(f"Runtime: {result['metadata']['runtime_seconds']}s")

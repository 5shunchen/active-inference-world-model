"""
Stage 4: Visual Renderer
Converts state trajectories to video frames and outputs MP4 video.

Uses matplotlib for frame generation and opencv for video encoding.
"""

import numpy as np
import os
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import io
import base64

# Try to import cv2, provide fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .tiger_ecology_model import TigerEcologyModel


class VisualRenderer:
    """Renders tiger simulation trajectories to video frames"""

    def __init__(self, model: TigerEcologyModel, width: int = 640, height: int = 480):
        self.model = model
        self.width = width
        self.height = height
        self.fps = 30

        # Colors
        self.colors = {
            'background': '#1a472a',
            'water': '#4a90d9',
            'prey_zone': '#8b7355',
            'safe_zone': '#2d8659',
            'tiger_full': '#ff6b35',
            'tiger_hungry': '#ff3333',
            'prey': '#f5e6c8',
            'tree': '#2d5a27',
        }

    def render_frame(self, state: int, step: int) -> np.ndarray:
        """Render a single frame for a given state"""
        state_info = self.model.analyze_state(state)

        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis('off')

        # Draw background zones based on position
        self._draw_zones(ax, state_info['position'])

        # Draw tiger
        self._draw_tiger(ax, state_info)

        # Draw UI/HUD
        self._draw_hud(ax, state_info, step)

        # Convert figure to numpy array
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        # Convert RGBA to BGR for opencv
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if CV2_AVAILABLE else frame

        return frame

    def _draw_zones(self, ax, position: str):
        """Draw the three environmental zones"""
        zone_width = self.width / 3

        # Prey zone (left)
        prey_color = self.colors['prey_zone'] if position == 'near_prey' else '#5a4a35'
        ax.add_patch(Rectangle((0, 0), zone_width, self.height, color=prey_color))
        ax.text(zone_width/2, self.height - 30, 'PREY', ha='center', va='center', color='white', fontweight='bold')

        # Water zone (middle)
        water_color = self.colors['water'] if position == 'near_water' else '#2a5080'
        ax.add_patch(Rectangle((zone_width, 0), zone_width, self.height, color=water_color))
        ax.text(zone_width*1.5, self.height - 30, 'WATER', ha='center', va='center', color='white', fontweight='bold')

        # Safe zone (right)
        safe_color = self.colors['safe_zone'] if position == 'safe_territory' else '#1a5a39'
        ax.add_patch(Rectangle((zone_width*2, 0), zone_width, self.height, color=safe_color))
        ax.text(zone_width*2.5, self.height - 30, 'SAFE', ha='center', va='center', color='white', fontweight='bold')

        # Draw decorative trees
        for i in range(3):
            tree_x = i * zone_width + zone_width/2
            self._draw_tree(ax, tree_x, 50)

    def _draw_tree(self, ax, x: float, y: float):
        """Draw a simple tree"""
        ax.add_patch(Rectangle((x-5, y), 10, 30, color='#5a3d2b'))
        ax.add_patch(Circle((x, y+30), 25, color=self.colors['tree']))

    def _draw_tiger(self, ax, state_info: Dict[str, str]):
        """Draw the tiger at current position"""
        zone_width = self.width / 3

        # Determine which zone the tiger is in
        if state_info['position'] == 'near_prey':
            x = zone_width * 0.5
        elif state_info['position'] == 'near_water':
            x = zone_width * 1.5
        else:  # safe_territory
            x = zone_width * 2.5

        y = self.height / 2

        # Tiger color based on hunger and health
        if state_info['hunger'] == 'starving':
            color = self.colors['tiger_hungry']
        elif state_info['hunger'] == 'full':
            color = self.colors['tiger_full']
        else:
            color = '#ff8c42'

        # Draw tiger body (simplified)
        ax.add_patch(Circle((x, y), 35, color=color, ec='black', lw=2))

        # Draw stripes
        for i in range(3):
            stripe_x = x - 15 + i * 15
            ax.plot([stripe_x-5, stripe_x+5], [y+10, y+10], 'black', lw=3)
            ax.plot([stripe_x-3, stripe_x+3], [y-10, y-10], 'black', lw=2)

        # Draw eyes
        ax.plot([x-10, x-10], [y+5, y+10], 'black', lw=3)
        ax.plot([x+10, x+10], [y+5, y+10], 'black', lw=3)

        # Draw age indicator
        age_text = state_info['age'].upper()
        ax.text(x, y - 50, age_text, ha='center', va='center', color='white', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.7, pad=2))

    def _draw_hud(self, ax, state_info: Dict[str, str], step: int):
        """Draw heads-up display with state info"""
        # Hunger bar
        ax.text(10, self.height - 60, f"HUNGER: {state_info['hunger'].upper()}", color='white', fontweight='bold')

        # Health bar
        health_color = '#00ff00' if state_info['health'] in ['healthy', 'excellent'] else '#ffff00' if state_info['health'] == 'healthy' else '#ff0000'
        ax.text(10, self.height - 80, f"HEALTH: {state_info['health'].upper()}", color=health_color, fontweight='bold')

        # Step counter
        ax.text(self.width - 80, self.height - 60, f"STEP: {step}", color='white', fontweight='bold')

    def render_trajectory(self, states: List[int]) -> List[np.ndarray]:
        """Render all frames for a trajectory"""
        frames = []
        for i, state in enumerate(states):
            frame = self.render_frame(state, i)
            frames.append(frame)
        return frames

    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = None) -> str:
        """Save frames as MP4 video"""
        if fps is None:
            fps = self.fps

        if not CV2_AVAILABLE:
            # Fallback: save as individual frames
            output_dir = output_path.replace('.mp4', '_frames')
            os.makedirs(output_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                import matplotlib.image as mpimg
                mpimg.imsave(f"{output_dir}/frame_{i:04d}.png", frame)
            print(f"OpenCV not available. Saved {len(frames)} frames to {output_dir}")
            return output_dir

        # Use opencv to write MP4
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved to {output_path}")
        return output_path

    def generate_video_from_trajectory(self, trajectory, output_path: str) -> str:
        """Generate video from WorldSimulator trajectory object"""
        frames = self.render_trajectory(trajectory.states)
        return self.save_video(frames, output_path)


def render_static_demo(output_path: str = "tiger_demo.mp4"):
    """Render a short demo video"""
    model = TigerEcologyModel()

    # Create sample trajectory through all positions
    sample_states = []
    for age in [0, 1, 2, 3]:
        for pos in [0, 1, 2]:
            state = model._encode_state(pos, 1, 1, age)
            sample_states.extend([state] * 10)  # Hold each state for 10 frames

    renderer = VisualRenderer(model)
    renderer.generate_video_from_trajectory(type('T', (), {'states': sample_states}), output_path)


if __name__ == "__main__":
    render_static_demo()

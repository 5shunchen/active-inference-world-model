"""
Stage 1: Thermostat Environment - A simple active inference demo
The thermostat maintains temperature by minimizing expected free energy.

States: [cold, comfortable, hot] (3 states)
Actions: [cool, do_nothing, heat] (3 actions)
Observations: [feel_cold, feel_comfortable, feel_hot] (3 observations)
Preferences: Prefer the comfortable state
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generative_model import DiscreteGenerativeModel
from src.world_simulator import WorldSimulator
from src.inference_engine import ActiveInferenceAgent, run_active_inference_loop


def create_thermostat_model(target_temp_idx: int = 1) -> DiscreteGenerativeModel:
    """
    Create a thermostat generative model.

    Args:
        target_temp_idx: Preferred state (0=cold, 1=comfortable, 2=hot)
    """
    n_states = 3  # cold, comfortable, hot
    n_observations = 3  # feel_cold, feel_comfortable, feel_hot
    n_actions = 3  # cool, do_nothing, heat

    # Observation model A: P(observation | state)
    # Mostly accurate observations with some noise
    A = np.eye(n_observations, n_states) * 0.9
    A += 0.05  # Add some noise
    A = A / np.sum(A, axis=0, keepdims=True)  # Normalize columns

    # Transition model B: P(next_state | current_state, action)
    B = np.zeros((n_states, n_states, n_actions))

    # Action 0: cool - temperature tends to decrease
    B[:, :, 0] = np.array([
        [0.8, 0.6, 0.1],  # next = cold
        [0.2, 0.3, 0.4],  # next = comfortable
        [0.0, 0.1, 0.5],  # next = hot
    ])

    # Action 1: do_nothing - temperature stays about same
    B[:, :, 1] = np.array([
        [0.8, 0.1, 0.0],  # next = cold
        [0.2, 0.8, 0.2],  # next = comfortable
        [0.0, 0.1, 0.8],  # next = hot
    ])

    # Action 2: heat - temperature tends to increase
    B[:, :, 2] = np.array([
        [0.5, 0.1, 0.0],  # next = cold
        [0.4, 0.3, 0.2],  # next = comfortable
        [0.1, 0.6, 0.8],  # next = hot
    ])

    # Preferences C: log probabilities over observations
    # Prefer comfortable state
    C = np.log(np.array([0.05, 0.9, 0.05]))
    C = C / np.sum(C)

    # Initial belief D: uniform
    D = np.array([1/3, 1/3, 1/3])

    return DiscreteGenerativeModel(n_states, n_observations, n_actions, A, B, C, D)


def run_thermostat_demo(n_steps: int = 50):
    """Run the thermostat demo and print results"""
    # Create model
    model = create_thermostat_model()

    # Create world (start in cold state)
    world = WorldSimulator(model, initial_state=0)

    # Create agent
    agent = ActiveInferenceAgent(model)

    # Run active inference loop
    run_active_inference_loop(agent, world, n_steps=n_steps)

    # Analyze results
    print(f"Thermostat simulation - {n_steps} steps")
    print("=" * 50)

    state_names = ["cold", "comfortable", "hot"]
    action_names = ["cool", "do_nothing", "heat"]

    # Count time in each state
    state_counts = np.bincount(world.trajectory.states, minlength=3)
    print(f"\nTime in each state:")
    for i, name in enumerate(state_names):
        print(f"  {name}: {state_counts[i]} steps ({state_counts[i]/n_steps*100:.1f}%)")

    # Count actions taken
    action_counts = np.bincount(world.trajectory.actions, minlength=3)
    print(f"\nActions taken:")
    for i, name in enumerate(action_names):
        print(f"  {name}: {action_counts[i]} times")

    # Free energy trend
    fe = world.trajectory.free_energy
    if fe:
        print(f"\nFree energy:")
        print(f"  Initial: {fe[0]:.4f}")
        print(f"  Final: {fe[-1]:.4f}")
        print(f"  Mean: {np.mean(fe):.4f}")

    return world, agent


if __name__ == "__main__":
    world, agent = run_thermostat_demo(100)

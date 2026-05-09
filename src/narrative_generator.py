"""
Stage 5: Narrative Generator
Detects key events from trajectory and generates time-aligned text descriptions.
Outputs JSON format for subtitles.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple

from .tiger_ecology_model import TigerEcologyModel


class NarrativeGenerator:
    """Generates natural language descriptions of tiger life trajectories"""

    def __init__(self, model: TigerEcologyModel):
        self.model = model

        # Event templates
        self.templates = {
            'birth': [
                "A new tiger cub is born in the safety of its territory.",
                "A tiger cub enters the world, beginning its journey through life.",
            ],
            'first_hunt': [
                "The young tiger attempts its first hunt in the prey-rich zone.",
                "Our tiger ventures out to learn the art of hunting.",
            ],
            'successful_hunt': [
                "The tiger successfully catches prey and enjoys a meal.",
                "A successful hunt! The tiger's hunger is satisfied.",
            ],
            'drinking': [
                "The tiger quenches its thirst at the water source.",
                "Our tiger stops to drink and refresh itself.",
            ],
            'resting': [
                "The tiger rests and recovers in the safe territory.",
                "Time for rest and recuperation in the safety zone.",
            ],
            'mating': [
                "The adult tiger seeks out a mate during breeding season.",
                "Our tiger is now an adult and ready to reproduce.",
            ],
            'injury': [
                "The tiger sustains an injury during a risky activity.",
                "An unfortunate incident leaves our tiger injured.",
            ],
            'healing': [
                "The tiger recovers and regains its strength.",
                "With rest, the tiger's health improves.",
            ],
            'aging': [
                "The tiger grows older and enters a new phase of life.",
                "Time passes, and our tiger matures further.",
            ],
            'death': [
                "The tiger's life journey comes to an end.",
                "After a full life, our tiger passes away.",
            ],
            'general': [
                "The tiger explores its environment.",
                "Another chapter unfolds in the tiger's life.",
            ]
        }

    def detect_events(self, states: List[int], actions: List[int]) -> List[Dict[str, Any]]:
        """Detect key events in the trajectory"""
        events = []

        # Track state changes
        prev_state_info = None

        for step, state in enumerate(states):
            state_info = self.model.analyze_state(state)

            # Birth event (first step)
            if step == 0:
                events.append({
                    'time': step,
                    'type': 'birth',
                    'state': state_info,
                })

            # Check for state transitions
            if prev_state_info is not None:
                # Position change
                if state_info['position'] != prev_state_info['position']:
                    if state_info['position'] == 'near_water':
                        events.append({
                            'time': step,
                            'type': 'drinking',
                            'state': state_info,
                        })
                    elif state_info['position'] == 'safe_territory' and state_info['health'] != 'injured':
                        events.append({
                            'time': step,
                            'type': 'resting',
                            'state': state_info,
                        })

                # Health change - injury
                if state_info['health'] == 'injured' and prev_state_info['health'] != 'injured':
                    events.append({
                        'time': step,
                        'type': 'injury',
                        'state': state_info,
                    })

                # Health change - healing
                if state_info['health'] in ['healthy', 'excellent'] and prev_state_info['health'] == 'injured':
                    events.append({
                        'time': step,
                        'type': 'healing',
                        'state': state_info,
                    })

                # Age change
                if state_info['age'] != prev_state_info['age']:
                    events.append({
                        'time': step,
                        'type': 'aging',
                        'state': state_info,
                    })

            # Check for hunt-related actions
            if step < len(actions) and actions[step] in [2]:  # hunt action
                if state_info['hunger'] == 'full' and step > 0:
                    events.append({
                        'time': step,
                        'type': 'successful_hunt',
                        'state': state_info,
                    })
                else:
                    events.append({
                        'time': step,
                        'type': 'first_hunt' if step < 20 else 'general',
                        'state': state_info,
                    })

            prev_state_info = state_info

        return events

    def generate_narrative(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate narrative text for detected events"""
        narrative = []

        seen_types = set()

        for event in events:
            event_type = event['type']

            # Select template
            templates = self.templates.get(event_type, self.templates['general'])

            # Vary template selection for variety
            template_idx = len(seen_types) % len(templates)
            text = templates[template_idx]

            # Add state details to make narrative richer
            state = event['state']
            details = f" (Age: {state['age']}, Health: {state['health']})"

            narrative.append({
                'time': event['time'],
                'text': text + details,
                'event_type': event_type,
            })

            seen_types.add(event_type)

        return narrative

    def generate_summary(self, states: List[int]) -> str:
        """Generate a summary of the entire life trajectory"""
        if not states:
            return "No life data available."

        # Analyze states
        final_state = self.model.analyze_state(states[-1])
        ages_seen = set()
        positions_visited = set()

        for state in states:
            info = self.model.analyze_state(state)
            ages_seen.add(info['age'])
            positions_visited.add(info['position'])

        summary = (
            f"Life Summary:\n"
            f"- Duration: {len(states)} simulation steps\n"
            f"- Final age: {final_state['age']}\n"
            f"- Age stages experienced: {', '.join(sorted(ages_seen))}\n"
            f"- Areas visited: {', '.join(sorted(positions_visited))}\n"
        )

        return summary

    def generate_subtitles_json(self, narrative: List[Dict[str, Any]], fps: int = 30) -> str:
        """Generate subtitle JSON file format for video integration"""
        subtitles = []

        for entry in narrative:
            # Convert step to timestamp
            start_time = entry['time'] / fps

            subtitles.append({
                'start_time': round(start_time, 2),
                'end_time': round(start_time + 2.0, 2),  # Show for 2 seconds
                'text': entry['text'],
            })

        return json.dumps(subtitles, indent=2, ensure_ascii=False)

    def process_trajectory(self, trajectory) -> Dict[str, Any]:
        """Process a complete trajectory and return all narrative outputs"""
        events = self.detect_events(trajectory.states, trajectory.actions)
        narrative = self.generate_narrative(events)
        summary = self.generate_summary(trajectory.states)
        subtitles_json = self.generate_subtitles_json(narrative)

        return {
            'events': events,
            'narrative': narrative,
            'summary': summary,
            'subtitles_json': subtitles_json,
        }


def generate_life_narrative(trajectory, model: TigerEcologyModel) -> Dict[str, Any]:
    """Convenience function to generate full narrative"""
    generator = NarrativeGenerator(model)
    return generator.process_trajectory(trajectory)

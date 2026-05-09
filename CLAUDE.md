# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Active Inference World Simulator** - an end-to-end system that generates life story videos from natural language prompts. For example, input "给我一个老虎的生命历程" (Give me a tiger's life journey) outputs:
- A video showing the tiger's complete life from birth to death
- Synchronized textual narration of key life events

The system is fully driven by active inference principles. Agents (like tigers) survive, grow, hunt, reproduce, and die by minimizing variational free energy in a simulated world.

## Architecture

The system is organized into 6 core modules with strict dependency ordering:

```
Stage 0: Generative Model Interfaces & Data Structures
↓
Stage 1: World Simulator + Inference Engine (Active Inference Loop)
↓
Stage 2: Language Parser & Scene Builder
↓
Stage 3: Agent Life Model
↓
Stage 4: Visual Renderer
↓
Stage 5: Narrative Generator
↓
Stage 6: End-to-End API
```

### Module Details

**Stage 0 - Foundation**
- Files: `generative_model.py`, `distributions.py`
- Defines `GenerativeModel` ABC with `get_observation_model()`, `get_transition_model()`, `get_preferences()`, `get_initial_belief()`
- Implements probability distributions: `Categorical`, `MultivariateGaussian` with `sample()` and `log_prob()`

**Stage 1 - Core Loop**
- Files: `world_simulator.py`, `inference_engine.py`
- `WorldSimulator`: Hides true state, exposes `step(action) -> obs` and `observe()`
- `InferenceEngine`: `infer_state()`, `compute_efe()`, `plan_action()`
- Discrete: Bayesian filtering, MCTS for planning
- Continuous: Extended Kalman/particle filtering

**Stage 2 - NLP to World**
- Files: `language_parser.py`, `scene_builder.py`, `scene_templates.json`
- Maps natural language prompts → structured scene JSON → instantiated `GenerativeModel`

**Stage 3 - Ecology**
- File: `tiger_ecology_model.py` (extends `GenerativeModel`)
- State factors: position, hunger, health, age, reproductive state
- Actions: move, drink, hunt, eat, rest, mate, nurture
- Dynamic preferences (C matrix changes with state/time)

**Stage 4 - Video Rendering**
- File: `visual_renderer.py`
- Converts state trajectories → video frames → MP4
- Supports resolution/framerate configuration

**Stage 5 - Narration**
- File: `narrative_generator.py`
- Detects key events from trajectories
- Generates time-aligned text descriptions
- Output: JSON `[{"time": 0, "text": "..."}]` for subtitles

**Stage 6 - API**
- File: `life_story_api.py`
- Entry point: `LifestoryAPI.generate(prompt, output_video_path) -> dict`

## Common Commands

*(Note: Actual code files don't exist yet. Commands will be updated as code is implemented)*

### Expected Project Structure
```
src/
  generative_model.py
  distributions.py
  world_simulator.py
  inference_engine.py
  language_parser.py
  scene_builder.py
  tiger_ecology_model.py
  visual_renderer.py
  narrative_generator.py
  life_story_api.py
tests/
  test_*.py
requirements.txt
```

### Development Workflow
1. **Iterate stage by stage** - Each stage must pass tests before advancing
2. **Stage 3 and 4 can be partially parallel** - Render placeholder video with simplified model first
3. **Stage 5**: Start with rule-based templates, then optionally add LLM polishing

### Testing
- Run all tests: `python -m pytest tests/ -v`
- Run single test file: `python -m pytest tests/test_thermostat.py -v`
- Run specific test: `python -m pytest tests/test_thermostat.py::test_thermostat_maintains_temp -v`

## Key Acceptance Criteria

| Stage | Validation |
|-------|------------|
| 0 | All distribution/model unit tests pass |
| 1 | Thermostat maintains temperature; free energy decreases over time |
| 2 | "老虎的生命历程" reliably maps to correct scene config |
| 3 | Multiple simulations show full life stages (birth → adulthood → reproduction → death) |
| 4 | Static trajectory produces continuous, playable video |
| 5 | Narrative covers birth, growth, hunting, reproduction, death |
| 6 | `generate("给我一个老虎的生命历程")` returns 2-5 min video + synchronized text |

## Development Guidelines

- Each stage should be independently testable
- Always run regression tests after completing each stage
- The `prompt.md` file is the authoritative design document
- No code exists yet - start with Stage 0 foundations

## Reference

See `prompt.md` for the full detailed design specification.

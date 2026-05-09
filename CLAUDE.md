# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Active Inference World Simulator** - an end-to-end simulation platform that generates life story videos from natural language prompts. Agents survive, grow, hunt, reproduce, and die by minimizing variational free energy in a dynamic ecosystem.

## Current Status

✅ **All 81 tests passing** | 13 modules | Full API & CLI support

## Architecture

The system is organized into 6 core stages plus new extensions:

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
↓
NEW:     Advanced Active Inference Engine
NEW:     Multi-Agent Ecosystem Simulator
NEW:     RESTful Web API (FastAPI)
NEW:     Command Line Interface (CLI)
```

## Core Modules

| Stage | File | Description |
|-------|------|-------------|
| 0 | `distributions.py` | Categorical, MultivariateGaussian distributions |
| 0 | `generative_model.py` | Abstract base class for generative models |
| 1 | `world_simulator.py` | Environment with state hiding, trajectory recording |
| 1 | `inference_engine.py` | Bayesian filtering, EFE computation, action planning |
| 1 | `thermostat.py` | Thermostat demo environment |
| 2 | `language_parser.py` | Natural language → scene config mapping |
| 2 | `scene_builder.py` | Builds generative models from scene configs |
| 2 | `scene_templates.json` | Scene templates and configuration |
| 3 | `tiger_ecology_model.py` | Full tiger lifecycle model |
| 4 | `visual_renderer.py` | Matplotlib frame rendering + MP4 generation |
| 5 | `narrative_generator.py` | Event detection + time-aligned text descriptions |
| 6 | `life_story_api.py` | `LifestoryAPI.generate()` with full pipeline integration |
| + | `active_inference.py` | Advanced EFE computation, policy optimization |
| + | `ecosystem.py` | Multi-agent predator-prey simulation with dynamic environment |
| + | `api.py` | FastAPI REST server with background tasks |
| + | `cli.py` | Command-line interface |

## New Features (v1.0)

### 🌍 Multi-Agent Ecosystem
- **Agents**: Tigers (predators), Deer (prey)
- **Environment**: 4 zones (prey, water, safe, grass)
- **Dynamics**: Day/night cycle, weather changes, resource regeneration
- **Interactions**: Predation, resource competition

### 🧠 Advanced Active Inference
- Proper Expected Free Energy (EFE) computation
- Risk + Ambiguity decomposition
- Bayesian filtering for state estimation
- Action selection via EFE minimization

### 🌐 Web API
- Framework: FastAPI
- Endpoints:
  - `POST /api/ecosystem` - Start ecosystem simulation
  - `POST /api/lifestory` - Generate life story from prompt
  - `GET /api/results/{id}` - Get simulation results
  - `GET /api/health` - Health check
  - `GET /api/config` - Configuration options
- Interactive docs: `/docs`

### 💻 Command Line Interface
```bash
# Run tiger life story
python -m src.cli run-tiger --steps 200 --video

# Run ecosystem simulation
python -m src.cli run-ecosystem --tigers 2 --deer 10

# Start API server
python -m src.cli serve --port 8000

# Run tests
python -m src.cli test -v
```

## Project Structure
```
active-inference-world-model/
├── src/
│   ├── distributions.py          # Stage 0
│   ├── generative_model.py       # Stage 0
│   ├── world_simulator.py        # Stage 1
│   ├── inference_engine.py       # Stage 1
│   ├── thermostat.py              # Stage 1
│   ├── language_parser.py         # Stage 2
│   ├── scene_builder.py           # Stage 2
│   ├── scene_templates.json       # Stage 2
│   ├── tiger_ecology_model.py     # Stage 3
│   ├── visual_renderer.py         # Stage 4
│   ├── narrative_generator.py     # Stage 5
│   ├── life_story_api.py          # Stage 6
│   ├── active_inference.py        # Advanced AI
│   ├── ecosystem.py               # Multi-agent sim
│   ├── api.py                     # FastAPI server
│   └── cli.py                     # CLI
├── tests/
│   ├── test_ecosystem.py
│   ├── test_stage0_distributions.py
│   ├── test_stage1_thermostat.py
│   ├── test_stage2_parser.py
│   ├── test_stage3_tiger.py
│   ├── test_stage4_renderer.py
│   ├── test_stage5_narrative.py
│   └── test_stage6_api.py
├── requirements.txt
├── CLAUDE.md
└── prompt.md
```

## Common Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Development
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_ecosystem.py -v

# Run single test
python -m pytest tests/test_ecosystem.py::TestEcosystemSimulator::test_full_simulation
```

### Usage
```bash
# CLI
python -m src.cli run-ecosystem --tigers 1 --deer 5 --steps 300
python -m src.cli serve

# Python API
from src.ecosystem import EcosystemSimulator
sim = EcosystemSimulator(n_tigers=2, n_deer=8)
history = sim.run(max_steps=500)
```

## Product Roadmap

### ✅ Phase 1 (Completed)
- Core 6-stage pipeline implementation
- Tiger life story generation
- Video rendering
- Narrative generation

### ✅ Phase 2 (Completed)
- Advanced active inference engine with proper EFE computation
- Multi-agent ecosystem (tiger + deer)
- Dynamic environment (day/night, weather, resources)
- Predator-prey interactions

### ✅ Phase 3 (Completed)
- FastAPI REST server
- Background task processing
- CLI interface
- 81 passing tests

### 🚧 Phase 4 (Next)
- Docker containerization
- Web dashboard with real-time visualization
- More agent types (wolf, rabbit, fox)
- Ecosystem complexity (plants, terrain, seasons)
- LLM integration for enhanced narratives
- Persistent storage database

## Reference

See `prompt.md` for the original design specification.

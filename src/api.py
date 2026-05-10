"""
RESTful API Service for Active Inference World Simulator

Endpoints:
- POST /api/simulate: Run a full simulation
- POST /api/ecosystem: Run multi-agent ecosystem simulation
- GET /api/health: Health check
- GET /api/config: Get available configurations
- POST /api/agent: Create custom agent
- GET /api/results/{id}: Get simulation results
"""

import os
import json
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ecosystem import EcosystemSimulator, AgentType
from .life_story_api import LifestoryAPI


app = FastAPI(
    title="Active Inference World Simulator API",
    description="End-to-end simulation platform for ecological life stories using active inference",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for simulation results
simulation_results: Dict[str, Dict] = {}

# Mount static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure directories exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# Pydantic Models
class AgentConfig(BaseModel):
    agent_type: str = "tiger"
    count: int = 1
    initial_position: Optional[int] = None


class EcosystemRequest(BaseModel):
    n_tigers: int = Field(default=1, ge=0, le=10, description="Number of tiger agents")
    n_wolves: int = Field(default=2, ge=0, le=15, description="Number of wolf agents")
    n_deer: int = Field(default=5, ge=0, le=30, description="Number of deer agents")
    n_foxes: int = Field(default=3, ge=0, le=15, description="Number of fox agents")
    n_rabbits: int = Field(default=8, ge=0, le=50, description="Number of rabbit agents")
    max_steps: int = Field(default=500, ge=1, le=5000, description="Maximum simulation steps")
    render_video: bool = Field(default=False, description="Generate video output")
    enhanced_mode: bool = Field(default=True, description="Use enhanced ecosystem with all species")

    class Config:
        schema_extra = {
            "example": {
                "n_tigers": 1,
                "n_wolves": 2,
                "n_deer": 5,
                "n_foxes": 3,
                "n_rabbits": 8,
                "max_steps": 300,
                "render_video": False,
                "enhanced_mode": True
            }
        }


class LifeStoryRequest(BaseModel):
    prompt: str = Field(default="给我一个老虎的生命历程", description="Natural language prompt for life story")
    max_steps: int = Field(default=500, ge=1, le=2000, description="Maximum simulation steps")
    render_video: bool = Field(default=False, description="Generate video output")


class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str
    created_at: str


class EcosystemResult(BaseModel):
    simulation_id: str
    total_steps: int
    surviving_tigers: int
    surviving_deer: int
    history: List[Dict]
    summary: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    available_modules: List[str]


# Background task runner
def run_ecosystem_simulation(
    simulation_id: str,
    n_tigers: int,
    n_wolves: int,
    n_deer: int,
    n_foxes: int,
    n_rabbits: int,
    max_steps: int,
    render_video: bool = False,
    enhanced_mode: bool = True
):
    """Run ecosystem simulation in background"""
    start_time = time.time()

    try:
        if enhanced_mode:
            from .enhanced_ecosystem import EnhancedEcosystemSimulator
            sim = EnhancedEcosystemSimulator(
                n_tigers=n_tigers,
                n_wolves=n_wolves,
                n_deer=n_deer,
                n_foxes=n_foxes,
                n_rabbits=n_rabbits
            )
        else:
            sim = EcosystemSimulator(n_tigers=n_tigers, n_deer=n_deer)
        history = sim.run(max_steps=max_steps)

        # Collect results
        final = history[-1] if history else {}

        results = {
            "simulation_id": simulation_id,
            "status": "completed",
            "total_steps": len(history),
            "surviving_tigers": final.get("tigers_alive", 0),
            "surviving_deer": final.get("deer_alive", 0),
            "history": history[-50:],  # Last 50 steps only
            "summary": sim.get_summary(),
            "runtime_seconds": round(time.time() - start_time, 2),
            "render_video": render_video,
            "video_path": None,
        }

        simulation_results[simulation_id] = results

    except Exception as e:
        simulation_results[simulation_id] = {
            "simulation_id": simulation_id,
            "status": "failed",
            "error": str(e),
            "runtime_seconds": round(time.time() - start_time, 2),
        }


def run_lifestory_simulation(
    simulation_id: str,
    prompt: str,
    max_steps: int,
    render_video: bool
):
    """Run life story simulation in background"""
    start_time = time.time()

    try:
        output_dir = f"output/{simulation_id}"
        os.makedirs(output_dir, exist_ok=True)
        video_path = os.path.join(output_dir, "lifestory.mp4") if render_video else None

        result = LifestoryAPI.generate(
            prompt=prompt,
            output_video_path=video_path or "",
            max_steps=max_steps,
            render_video=render_video
        )

        results = {
            "simulation_id": simulation_id,
            "status": "completed",
            "prompt": prompt,
            "result": result,
            "runtime_seconds": round(time.time() - start_time, 2),
        }

        simulation_results[simulation_id] = results

    except Exception as e:
        simulation_results[simulation_id] = {
            "simulation_id": simulation_id,
            "status": "failed",
            "error": str(e),
            "runtime_seconds": round(time.time() - start_time, 2),
        }


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Web Dashboard - main page"""
    dashboard_path = os.path.join(TEMPLATE_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Active Inference World Simulator</h1><p>Dashboard coming soon</p>")


@app.get("/api", response_model=Dict[str, str])
async def api_root():
    """API root endpoint with information"""
    return {
        "name": "Active Inference World Simulator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "dashboard": "/",
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        available_modules=[
            "active_inference",
            "ecosystem",
            "lifestory",
            "visual_renderer",
            "narrative_generator"
        ]
    )


@app.post("/api/ecosystem", response_model=SimulationResponse)
async def start_ecosystem_simulation(
    request: EcosystemRequest,
    background_tasks: BackgroundTasks
):
    """Start a new multi-agent ecosystem simulation"""
    simulation_id = str(uuid.uuid4())[:8]

    background_tasks.add_task(
        run_ecosystem_simulation,
        simulation_id=simulation_id,
        n_tigers=request.n_tigers,
        n_wolves=request.n_wolves,
        n_deer=request.n_deer,
        n_foxes=request.n_foxes,
        n_rabbits=request.n_rabbits,
        max_steps=request.max_steps,
        render_video=request.render_video,
        enhanced_mode=request.enhanced_mode
    )

    total_animals = request.n_tigers + request.n_wolves + request.n_deer + request.n_foxes + request.n_rabbits
    return SimulationResponse(
        simulation_id=simulation_id,
        status="running",
        message=f"Ecosystem simulation started with {total_animals} animals "
                f"(T:{request.n_tigers} W:{request.n_wolves} D:{request.n_deer} F:{request.n_foxes} R:{request.n_rabbits})",
        created_at=datetime.now().isoformat()
    )


@app.post("/api/lifestory", response_model=SimulationResponse)
async def start_lifestory_simulation(
    request: LifeStoryRequest,
    background_tasks: BackgroundTasks
):
    """Generate a life story video from natural language prompt"""
    simulation_id = str(uuid.uuid4())[:8]

    background_tasks.add_task(
        run_lifestory_simulation,
        simulation_id=simulation_id,
        prompt=request.prompt,
        max_steps=request.max_steps,
        render_video=request.render_video
    )

    return SimulationResponse(
        simulation_id=simulation_id,
        status="running",
        message=f"Life story simulation started for prompt: {request.prompt}",
        created_at=datetime.now().isoformat()
    )


@app.get("/api/results/{simulation_id}")
async def get_simulation_results(simulation_id: str):
    """Get results for a specific simulation"""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return simulation_results[simulation_id]


@app.get("/api/results")
async def list_all_simulations():
    """List all simulation IDs and their status"""
    return {
        "total_simulations": len(simulation_results),
        "simulations": [
            {
                "id": sid,
                "status": result.get("status", "unknown"),
                "runtime": result.get("runtime_seconds", 0)
            }
            for sid, result in simulation_results.items()
        ]
    }


@app.get("/api/config")
async def get_configuration():
    """Get available configuration options"""
    return {
        "agent_types": ["tiger", "deer"],
        "agent_actions": ["move_to_prey", "move_to_water", "move_to_safe", "move_to_grass", "eat", "drink", "rest", "hunt", "flee"],
        "environment_zones": ["prey_zone", "water_zone", "safe_zone", "grass_zone"],
        "weather_types": ["clear", "cloudy", "rain", "fog"],
        "max_agents_per_simulation": 30,
        "max_simulation_steps": 5000,
    }


@app.delete("/api/results/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Delete simulation results"""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")

    del simulation_results[simulation_id]
    return {"status": "success", "message": f"Simulation {simulation_id} deleted"}


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    import uvicorn
    print(f"🚀 Starting Active Inference World Simulator API on http://{host}:{port}")
    print(f"📚 Documentation available at http://{host}:{port}/docs")
    uvicorn.run(f"{__name__}:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()

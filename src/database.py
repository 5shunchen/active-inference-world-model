"""
Database persistence layer for simulation storage and replay

Uses SQLAlchemy ORM with SQLite for file-based storage
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Database setup
Base = declarative_base()


class Simulation(Base):
    """Stores simulation metadata and configuration"""
    __tablename__ = "simulations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(16), unique=True, index=True)
    name = Column(String(100), default="Unnamed Simulation")
    type = Column(String(32))  # ecosystem, lifestory, etc.
    status = Column(String(16))  # running, completed, failed
    max_steps = Column(Integer)
    total_steps = Column(Integer, default=0)
    runtime_seconds = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    config = Column(Text)  # JSON configuration
    summary = Column(Text)  # Final summary text

    # Relationship to steps
    steps = relationship("SimulationStep", back_populates="simulation", cascade="all, delete-orphan")


class SimulationStep(Base):
    """Stores individual step data for replay and analysis"""
    __tablename__ = "simulation_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    step_number = Column(Integer, index=True)

    # Environment state
    day_time = Column(Boolean)
    weather = Column(String(32))

    # Population counts
    tigers_alive = Column(Integer, default=0)
    wolves_alive = Column(Integer, default=0)
    deer_alive = Column(Integer, default=0)
    foxes_alive = Column(Integer, default=0)
    rabbits_alive = Column(Integer, default=0)
    total_alive = Column(Integer, default=0)

    # Resource levels
    water_level = Column(Float, default=0.0)
    grass_level = Column(Float, default=0.0)
    prey_level = Column(Float, default=0.0)

    # Relationship back to simulation
    simulation = relationship("Simulation", back_populates="steps")


class SimulationDatabase:
    """Database interface for simulation storage"""

    def __init__(self, db_path: str = "sqlite:///simulations.db"):
        self.engine = create_engine(db_path, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def create_simulation(
        self,
        simulation_id: str,
        sim_type: str,
        max_steps: int,
        config: Dict[str, Any] = None,
        name: str = None
    ) -> int:
        """Create a new simulation record"""
        with self.get_session() as session:
            sim = Simulation(
                simulation_id=simulation_id,
                name=name or f"Simulation {simulation_id}",
                type=sim_type,
                status="running",
                max_steps=max_steps,
                config=json.dumps(config or {}),
            )
            session.add(sim)
            session.commit()
            return sim.id

    def add_step(self, simulation_db_id: int, step_number: int, step_data: Dict[str, Any]):
        """Add a step record to a running simulation"""
        with self.get_session() as session:
            step = SimulationStep(
                simulation_id=simulation_db_id,
                step_number=step_number,
                day_time=step_data.get("day_time", True),
                weather=step_data.get("weather", "clear"),
                tigers_alive=step_data.get("tigers_alive", 0),
                wolves_alive=step_data.get("wolves_alive", 0),
                deer_alive=step_data.get("deer_alive", 0),
                foxes_alive=step_data.get("foxes_alive", 0),
                rabbits_alive=step_data.get("rabbits_alive", 0),
                total_alive=step_data.get("alive", 0),
                water_level=step_data.get("resources", {}).get("water", 0),
                grass_level=step_data.get("resources", {}).get("grass", 0),
                prey_level=step_data.get("resources", {}).get("prey_spawn", 0),
            )
            session.add(step)
            session.commit()

    def complete_simulation(
        self,
        simulation_db_id: int,
        total_steps: int,
        runtime_seconds: float,
        summary: str = None
    ):
        """Mark simulation as completed"""
        with self.get_session() as session:
            sim = session.query(Simulation).filter(Simulation.id == simulation_db_id).first()
            if sim:
                sim.status = "completed"
                sim.total_steps = total_steps
                sim.runtime_seconds = runtime_seconds
                sim.completed_at = datetime.utcnow()
                sim.summary = summary
                session.commit()

    def fail_simulation(self, simulation_db_id: int, error: str):
        """Mark simulation as failed"""
        with self.get_session() as session:
            sim = session.query(Simulation).filter(Simulation.id == simulation_db_id).first()
            if sim:
                sim.status = "failed"
                sim.summary = error
                session.commit()

    def get_simulation(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation metadata by public ID"""
        with self.get_session() as session:
            sim = session.query(Simulation).filter(Simulation.simulation_id == simulation_id).first()
            if not sim:
                return None

            return {
                "id": sim.simulation_id,
                "name": sim.name,
                "type": sim.type,
                "status": sim.status,
                "max_steps": sim.max_steps,
                "total_steps": sim.total_steps,
                "runtime_seconds": sim.runtime_seconds,
                "created_at": sim.created_at.isoformat(),
                "completed_at": sim.completed_at.isoformat() if sim.completed_at else None,
                "config": json.loads(sim.config),
                "summary": sim.summary,
            }

    def get_simulation_steps(
        self,
        simulation_id: str,
        limit: int = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get step data for a simulation"""
        with self.get_session() as session:
            sim = session.query(Simulation).filter(Simulation.simulation_id == simulation_id).first()
            if not sim:
                return []

            query = (
                session.query(SimulationStep)
                .filter(SimulationStep.simulation_id == sim.id)
                .order_by(SimulationStep.step_number)
                .offset(offset)
            )

            if limit:
                query = query.limit(limit)

            steps = query.all()

            return [
                {
                    "step": s.step_number,
                    "day_time": s.day_time,
                    "weather": s.weather,
                    "tigers": s.tigers_alive,
                    "wolves": s.wolves_alive,
                    "deer": s.deer_alive,
                    "foxes": s.foxes_alive,
                    "rabbits": s.rabbits_alive,
                    "total": s.total_alive,
                    "resources": {
                        "water": s.water_level,
                        "grass": s.grass_level,
                        "prey": s.prey_level,
                    },
                }
                for s in steps
            ]

    def list_simulations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all simulations in reverse chronological order"""
        with self.get_session() as session:
            sims = (
                session.query(Simulation)
                .order_by(Simulation.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": sim.simulation_id,
                    "name": sim.name,
                    "type": sim.type,
                    "status": sim.status,
                    "total_steps": sim.total_steps,
                    "runtime_seconds": sim.runtime_seconds,
                    "created_at": sim.created_at.isoformat(),
                }
                for sim in sims
            ]

    def delete_simulation(self, simulation_id: str) -> bool:
        """Delete a simulation and all its steps"""
        with self.get_session() as session:
            sim = session.query(Simulation).filter(Simulation.simulation_id == simulation_id).first()
            if not sim:
                return False

            session.delete(sim)
            session.commit()
            return True

    def get_population_timeseries(self, simulation_id: str) -> Dict[str, List]:
        """Get population data formatted for chart visualization"""
        steps = self.get_simulation_steps(simulation_id)

        return {
            "labels": [s["step"] for s in steps],
            "tigers": [s["tigers"] for s in steps],
            "wolves": [s["wolves"] for s in steps],
            "deer": [s["deer"] for s in steps],
            "foxes": [s["foxes"] for s in steps],
            "rabbits": [s["rabbits"] for s in steps],
        }


# Global database instance
db = SimulationDatabase()

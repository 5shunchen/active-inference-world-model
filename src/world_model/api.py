"""
World Model API - 通用世界模型 REST API

提供统一的接口来模拟演化任何事物
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .core import WorldModel, State, Action
from .registry import registry

# 加载领域模型
from .domains import ecology, economy

router = APIRouter(prefix="/world-model", tags=["World Model"])


class InitializeRequest(BaseModel):
    """初始化世界模型请求"""
    model_id: str
    config: Dict[str, Any]


class StepRequest(BaseModel):
    """推进模拟请求"""
    instance_id: str
    steps: int = 1
    actions: Optional[List[Dict[str, Any]]] = None


class RolloutRequest(BaseModel):
    """前向滚动模拟请求"""
    instance_id: str
    steps: int
    initial_state: Optional[Dict[str, Any]] = None


class CreateEntityRequest(BaseModel):
    """创建实体请求"""
    instance_id: str
    name: str
    entity_type: str = "agent"
    attributes: Dict[str, Any] = {}


@router.get("/models")
async def list_available_models():
    """列出所有可用的世界模型"""
    models = registry.list_models()
    return {
        "total": len(models),
        "models": models
    }


@router.get("/instances")
async def list_running_instances():
    """列出所有运行中的世界模型实例"""
    instances = registry.list_instances()
    return {
        "total": len(instances),
        "instances": instances
    }


@router.post("/instances")
async def create_model_instance(request: InitializeRequest):
    """
    创建并初始化一个世界模型实例

    Args:
        model_id: 模型ID，格式为 "领域/名称" (如 "ecology/predator_prey")
        config: 领域特定的配置参数

    可用模型:
        - ecology/predator_prey: 捕食者-猎物生态系统
        - ecology/simple_2_species: 简化双物种模型
        - economy/market_simulation: 市场经济模拟
    """
    model = registry.create_instance(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    model.initialize(request.config)

    return {
        "instance_id": model.model_id,
        "model_id": request.model_id,
        "status": "initialized",
        "statistics": model.get_statistics()
    }


@router.get("/instances/{instance_id}")
async def get_instance_state(instance_id: str):
    """获取模型实例的当前状态"""
    instance = registry.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    return instance.get_state_snapshot()


@router.get("/instances/{instance_id}/statistics")
async def get_instance_statistics(instance_id: str):
    """获取模型实例的统计信息"""
    instance = registry.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    return instance.get_statistics()


@router.post("/instances/{instance_id}/step")
async def step_simulation(instance_id: str, steps: int = 1):
    """
    推进模拟 N 步

    Args:
        instance_id: 模型实例ID
        steps: 推进步数，默认1步
    """
    instance = registry.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    states = []
    for _ in range(steps):
        new_state = instance.step()
        states.append({
            "timestep": new_state.timestep,
            "entity_count": len(new_state.entities)
        })

    return {
        "status": "completed",
        "steps_executed": steps,
        "final_timestep": instance.state.timestep,
        "states": states,
        "statistics": instance.get_statistics()
    }


@router.post("/instances/{instance_id}/rollout")
async def rollout_simulation(instance_id: str, steps: int):
    """
    执行前向滚动模拟（不修改当前状态）

    Args:
        instance_id: 模型实例ID
        steps: 滚动步数
    """
    instance = registry.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    trajectory = instance.rollout(steps)

    return {
        "status": "completed",
        "steps": steps,
        "trajectory_length": len(trajectory),
        "final_state": {
            "timestep": trajectory[-1].timestep,
            "entities": [
                {"id": e.entity_id, "name": e.name, "active": e.active}
                for e in trajectory[-1].entities.values()
            ]
        }
    }


@router.post("/instances/{instance_id}/entities")
async def create_entity(instance_id: str, request: CreateEntityRequest):
    """在世界中创建新实体"""
    instance = registry.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    from .core import EntityType

    type_map = {
        "agent": EntityType.AGENT,
        "object": EntityType.OBJECT,
        "resource": EntityType.RESOURCE,
        "concept": EntityType.CONCEPT,
        "relation": EntityType.RELATION,
        "event": EntityType.EVENT
    }

    entity_type = type_map.get(request.entity_type, EntityType.OBJECT)

    entity = instance.create_entity(
        name=request.name,
        entity_type=entity_type,
        attributes=request.attributes
    )

    return {
        "entity_id": entity.entity_id,
        "name": entity.name,
        "type": entity.entity_type.value,
        "attributes": request.attributes
    }


@router.delete("/instances/{instance_id}")
async def destroy_instance(instance_id: str):
    """销毁模型实例"""
    success = registry.destroy_instance(instance_id)
    if not success:
        raise HTTPException(status_code=404, detail="Instance not found")

    return {
        "status": "success",
        "message": f"Instance {instance_id} destroyed"
    }


@router.get("/domains")
async def get_available_domains():
    """获取所有可用的领域分类"""
    models = registry.list_models()
    domains = {}
    for model in models:
        domain = model["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append({
            "name": model["name"],
            "description": model["description"],
            "version": model["version"],
            "tags": model["tags"]
        })

    return {
        "domains": domains,
        "total_domains": len(domains)
    }


# 便捷的预设配置模板
PRESET_CONFIGS = {
    "ecology/simple": {
        "model_id": "ecology/simple_2_species",
        "config": {"predators": 10, "prey": 50, "resources": 0.8}
    },
    "ecology/complex": {
        "model_id": "ecology/predator_prey",
        "config": {
            "species": [
                {"name": "tigers", "type": "predator", "population": 5, "predation_rate": 0.1},
                {"name": "wolves", "type": "predator", "population": 8, "predation_rate": 0.08},
                {"name": "deer", "type": "prey", "population": 50, "reproduction_rate": 0.08},
                {"name": "rabbits", "type": "prey", "population": 80, "reproduction_rate": 0.12}
            ],
            "environment": {"resource_availability": 0.7}
        }
    },
    "economy/simple": {
        "model_id": "economy/market_simulation",
        "config": {
            "companies": [
                {"name": "Company_A", "cash": 10000, "production": 50, "efficiency": 0.9},
                {"name": "Company_B", "cash": 8000, "production": 40, "efficiency": 0.85}
            ],
            "resources": [
                {"name": "gold", "price": 100, "volatility": 0.05},
                {"name": "oil", "price": 50, "volatility": 0.1}
            ]
        }
    }
}


@router.get("/presets")
async def get_preset_configs():
    """获取预设的模拟配置模板"""
    return {
        "presets": list(PRESET_CONFIGS.keys()),
        "configs": PRESET_CONFIGS
    }


@router.post("/presets/{preset_id}/launch")
async def launch_preset_simulation(preset_id: str):
    """使用预设配置快速启动模拟"""
    if preset_id not in PRESET_CONFIGS:
        raise HTTPException(status_code=404, detail="Preset not found")

    preset = PRESET_CONFIGS[preset_id]

    model = registry.create_instance(preset["model_id"])
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    model.initialize(preset["config"])

    return {
        "instance_id": model.model_id,
        "preset": preset_id,
        "status": "launched",
        "statistics": model.get_statistics()
    }

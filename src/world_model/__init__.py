"""
World Model - 通用世界模型引擎

一个可扩展的世界建模框架，可以模拟演化任何事物。

核心组件：
- WorldModel: 世界模型基类，所有领域模型都继承自它
- Entity: 世界中的通用实体
- State: 世界状态的通用表示
- DynamicsModel: 演化动力学模型
- PerceptionModel: 感知模型
- DecisionModel: 决策模型
- registry: 世界模型注册表，管理可用的领域模型

可用领域模型：
- ecology/predator_prey: 生态系统捕食者-猎物模型
- ecology/simple_2_species: 简化双物种模型
- economy/market_simulation: 市场经济模拟
"""

from .core import (
    WorldModel,
    Entity,
    EntityType,
    State,
    Action,
    Perception,
    Attribute,
    DynamicsModel,
    PerceptionModel,
    DecisionModel,
    RewardModel
)

from .registry import (
    WorldModelRegistry,
    ModelMetadata,
    register_world_model,
    registry,
    load_models_from_module
)

# 自动加载内置领域模型
try:
    from .domains import ecology, economy
except ImportError:
    pass

__version__ = "3.0.0"
__all__ = [
    # Core
    "WorldModel",
    "Entity",
    "EntityType",
    "State",
    "Action",
    "Perception",
    "Attribute",
    "DynamicsModel",
    "PerceptionModel",
    "DecisionModel",
    "RewardModel",
    # Registry
    "WorldModelRegistry",
    "ModelMetadata",
    "register_world_model",
    "registry",
    "load_models_from_module",
]

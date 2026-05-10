"""
World Model Core - 通用世界模型核心引擎

这是一个通用的世界建模框架，可以模拟演化任何事物。
核心抽象：
- WorldModel: 世界模型基类，任何领域都可以继承实现
- Entity: 世界中的实体（可以是动物、植物、智能体、概念等）
- State: 世界状态的通用表示
- Action: 影响世界的动作
- Perception: 感知世界的接口
- Dynamics: 世界演化的动力学模型
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
import uuid
import time


class EntityType(Enum):
    """通用实体类型 - 可扩展"""
    AGENT = "agent"           # 智能体（主动决策）
    OBJECT = "object"         # 被动对象
    RESOURCE = "resource"     # 资源
    CONCEPT = "concept"       # 抽象概念
    RELATION = "relation"     # 关系
    EVENT = "event"           # 事件


@dataclass
class Attribute:
    """通用属性 - 任何实体都可以有任意属性"""
    name: str
    value: Any
    data_type: str = "float"  # float, int, string, bool, array, dict
    mutable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Entity:
    """世界中的通用实体"""
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: Dict[str, Attribute] = field(default_factory=dict)
    relations: List[str] = field(default_factory=list)  # 关联的其他实体ID
    active: bool = True
    created_at: int = field(default_factory=lambda: int(time.time()))

    def get(self, attr_name: str, default: Any = None) -> Any:
        """获取属性值"""
        attr = self.attributes.get(attr_name)
        return attr.value if attr else default

    def set(self, attr_name: str, value: Any) -> None:
        """设置属性值"""
        if attr_name in self.attributes:
            attr = self.attributes[attr_name]
            if attr.mutable:
                attr.value = value
        else:
            self.attributes[attr_name] = Attribute(attr_name, value)


class State:
    """世界状态的通用表示"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.global_attributes: Dict[str, Any] = {}
        self.timestep: int = 0

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_entities_by_attribute(self, attr_name: str, attr_value: Any) -> List[Entity]:
        return [e for e in self.entities.values() if e.get(attr_name) == attr_value]

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.entities:
            self.entities[entity_id].active = False

    def clone(self) -> 'State':
        """深度克隆状态"""
        import copy
        return copy.deepcopy(self)


@dataclass
class Action:
    """影响世界的动作"""
    action_id: str
    action_type: str
    actor_id: str
    target_ids: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Perception:
    """感知结果"""
    perceiver_id: str
    perceived_entities: List[str] = field(default_factory=list)
    perceived_attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class DynamicsModel(ABC):
    """世界演化的动力学模型 - 抽象基类"""

    @abstractmethod
    def predict_next_state(self, current_state: State, actions: List[Action]) -> State:
        """预测下一个状态"""
        pass


class RewardModel(ABC):
    """奖励/价值模型 - 用于评估世界状态"""

    @abstractmethod
    def evaluate(self, state: State) -> float:
        """评估给定状态的价值"""
        pass


class PerceptionModel(ABC):
    """感知模型 - 决定实体如何感知世界"""

    @abstractmethod
    def perceive(self, entity: Entity, state: State) -> Perception:
        """让实体感知世界状态"""
        pass


class DecisionModel(ABC):
    """决策模型 - 决定智能体如何选择动作"""

    @abstractmethod
    def decide(self, entity: Entity, perception: Perception, state: State) -> List[Action]:
        """基于感知和当前状态决定动作"""
        pass


class WorldModel(ABC):
    """
    通用世界模型基类

    任何领域的世界模型都应该继承这个类并实现核心方法：
    - initialize(): 初始化世界
    - step(): 推进一个时间步
    """

    def __init__(self, model_name: str, domain: str):
        self.model_name = model_name
        self.domain = domain
        self.model_id = str(uuid.uuid4())[:8]
        self.state = State()
        self.history: List[State] = []
        self.dynamics_models: Dict[str, DynamicsModel] = {}
        self.reward_models: Dict[str, RewardModel] = {}
        self.perception_models: Dict[str, PerceptionModel] = {}
        self.decision_models: Dict[str, DecisionModel] = {}
        self.initialized = False

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化世界模型

        Args:
            config: 初始化配置，可以包含任意领域特定的参数
        """
        self.initialized = True

    @abstractmethod
    def step(self, actions: Optional[List[Action]] = None) -> State:
        """
        推进世界演化一个时间步

        Args:
            actions: 外部或内部生成的动作列表

        Returns:
            新的世界状态
        """
        self.state.timestep += 1
        self.history.append(self.state.clone())
        return self.state

    def register_dynamics(self, name: str, model: DynamicsModel) -> None:
        """注册动力学模型"""
        self.dynamics_models[name] = model

    def register_reward(self, name: str, model: RewardModel) -> None:
        """注册奖励模型"""
        self.reward_models[name] = model

    def register_perception(self, name: str, model: PerceptionModel) -> None:
        """注册感知模型"""
        self.perception_models[name] = model

    def register_decision(self, name: str, model: DecisionModel) -> None:
        """注册决策模型"""
        self.decision_models[name] = model

    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取当前状态的序列化快照"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "domain": self.domain,
            "timestep": self.state.timestep,
            "entity_count": len(self.state.entities),
            "global_attributes": self.state.global_attributes,
            "entities": [
                {
                    "id": e.entity_id,
                    "type": e.entity_type.value,
                    "name": e.name,
                    "attributes": {k: v.value for k, v in e.attributes.items()}
                }
                for e in self.state.entities.values()
            ]
        }

    def rollout(self, steps: int, initial_state: Optional[State] = None) -> List[State]:
        """
        执行前向滚动模拟

        Args:
            steps: 模拟步数
            initial_state: 可选的初始状态

        Returns:
            状态历史列表
        """
        if initial_state:
            original_state = self.state
            self.state = initial_state.clone()

        trajectory = []
        for _ in range(steps):
            new_state = self.step()
            trajectory.append(new_state.clone())

        if initial_state:
            self.state = original_state

        return trajectory

    def create_entity(
        self,
        name: str,
        entity_type: EntityType,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Entity:
        """创建新实体"""
        entity_id = str(uuid.uuid4())[:8]
        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name
        )

        if attributes:
            for attr_name, attr_value in attributes.items():
                entity.set(attr_name, attr_value)

        self.state.add_entity(entity)
        return entity

    def get_statistics(self) -> Dict[str, Any]:
        """获取世界统计信息"""
        type_counts = {}
        for entity in self.state.entities.values():
            type_name = entity.entity_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "model_name": self.model_name,
            "domain": self.domain,
            "timestep": self.state.timestep,
            "total_entities": len(self.state.entities),
            "active_entities": sum(1 for e in self.state.entities.values() if e.active),
            "entity_types": type_counts,
            "history_length": len(self.history),
            "dynamics_models": list(self.dynamics_models.keys()),
            "reward_models": list(self.reward_models.keys()),
            "perception_models": list(self.perception_models.keys()),
            "decision_models": list(self.decision_models.keys())
        }

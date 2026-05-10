"""
Ecology Domain World Model - 生态领域世界模型

作为通用世界模型的一个领域插件示例
"""

import random
from typing import Dict, List, Any, Optional

from ..core import (
    WorldModel, EntityType, Action, State, DynamicsModel,
    PerceptionModel, DecisionModel, Perception
)
from ..registry import register_world_model


class EcologyDynamics(DynamicsModel):
    """生态系统动力学模型"""

    def predict_next_state(self, current_state: State, actions: List[Action]) -> State:
        new_state = current_state.clone()

        # 种群增长/衰减
        for entity in new_state.entities.values():
            if not entity.active:
                continue

            # 繁殖率
            reproduction_rate = entity.get("reproduction_rate", 0.01)
            population = entity.get("population", 0)

            # 承载力
            carrying_capacity = entity.get("carrying_capacity", 1000)

            # 逻辑斯谛增长
            if population > 0 and carrying_capacity > 0:
                growth = reproduction_rate * population * (1 - population / carrying_capacity)
                entity.set("population", max(0, int(population + growth)))

            # 能量消耗
            energy = entity.get("energy", 100.0)
            energy_decay = entity.get("energy_decay", 0.02)
            entity.set("energy", max(0, energy - energy_decay))

            # 死亡条件
            if entity.get("energy", 0) <= 0:
                entity.set("population", max(0, entity.get("population", 1) - 1))

            # 如果种群归零，标记为不活跃
            if entity.get("population", 0) <= 0:
                entity.active = False

        # 捕食交互
        predators = [e for e in new_state.entities.values() if e.get("is_predator", False)]
        prey = [e for e in new_state.entities.values() if e.get("is_prey", False)]

        for predator in predators:
            if not predator.active:
                continue

            predator_pop = predator.get("population", 0)
            if predator_pop <= 0:
                continue

            for prey_entity in prey:
                if not prey_entity.active:
                    continue

                prey_pop = prey_entity.get("population", 0)
                if prey_pop <= 0:
                    continue

                # 捕食率
                predation_rate = predator.get("predation_rate", 0.05)
                caught = int(prey_pop * predation_rate * random.uniform(0.5, 1.5))

                # 猎物减少
                prey_entity.set("population", max(0, prey_pop - caught))

                # 捕食者获得能量
                energy_gain = caught * prey_entity.get("nutrition_value", 1.0)
                current_energy = predator.get("energy", 100.0)
                predator.set("energy", current_energy + energy_gain)

        return new_state


class EcologyPerception(PerceptionModel):
    """生态系统感知模型"""

    def perceive(self, entity, state) -> Perception:
        perceived = []

        # 感知附近的其他实体
        for other in state.entities.values():
            if other.entity_id != entity.entity_id and other.active:
                distance = abs(entity.get("x", 0) - other.get("x", 0))
                if distance < entity.get("perception_range", 5):
                    perceived.append(other.entity_id)

        return Perception(
            perceiver_id=entity.entity_id,
            perceived_entities=perceived,
            confidence=1.0
        )


class EcologyDecision(DecisionModel):
    """生态系统决策模型"""

    def decide(self, entity, perception, state) -> List[Action]:
        actions = []

        # 简单的决策逻辑：能量低时移动向食物
        energy = entity.get("energy", 100)
        if energy < 30:
            for target_id in perception.perceived_entities:
                target = state.get_entity(target_id)
                if target and target.get("is_food", False):
                    actions.append(Action(
                        action_id=f"move_to_{target_id}",
                        action_type="move",
                        actor_id=entity.entity_id,
                        target_ids=[target_id],
                        parameters={"speed": 1.0}
                    ))
                    break

        return actions


@register_world_model(
    name="predator_prey",
    domain="ecology",
    description="通用捕食者-猎物生态系统模型，支持任意物种配置",
    version="3.0.0",
    author="Active Inference Team",
    tags=["ecology", "population", "predator-prey", "simulation"]
)
class EcologyWorldModel(WorldModel):
    """生态领域世界模型 - 作为通用世界模型的示例实现"""

    def __init__(self, model_name: str = "predator_prey", domain: str = "ecology"):
        super().__init__(model_name, domain)
        self.register_dynamics("population", EcologyDynamics())
        self.register_perception("basic", EcologyPerception())
        self.register_decision("simple", EcologyDecision())

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化生态系统

        Args:
            config: 配置示例
            {
                "species": [
                    {"name": "tigers", "type": "predator", "population": 5, ...},
                    {"name": "deer", "type": "prey", "population": 50, ...}
                ],
                "environment": {"resource_availability": 0.8}
            }
        """
        super().initialize(config)

        # 全局环境属性
        self.state.global_attributes.update({
            "resource_availability": config.get("environment", {}).get("resource_availability", 0.5),
            "season": config.get("environment", {}).get("season", "spring"),
            "weather": config.get("environment", {}).get("weather", "clear")
        })

        # 创建物种实体
        for species_config in config.get("species", []):
            species_type = species_config.get("type", "prey")

            attributes = {
                "population": species_config.get("population", 10),
                "reproduction_rate": species_config.get("reproduction_rate", 0.05),
                "carrying_capacity": species_config.get("carrying_capacity", 500),
                "energy": species_config.get("initial_energy", 100.0),
                "energy_decay": species_config.get("energy_decay", 0.02),
                "is_predator": species_type == "predator",
                "is_prey": species_type == "prey",
                "predation_rate": species_config.get("predation_rate", 0.05),
                "nutrition_value": species_config.get("nutrition_value", 1.0),
                "x": random.uniform(0, 10),
                "y": random.uniform(0, 10),
                "perception_range": species_config.get("perception_range", 5.0)
            }

            self.create_entity(
                name=species_config["name"],
                entity_type=EntityType.AGENT,
                attributes=attributes
            )

    def step(self, actions: Optional[List[Action]] = None) -> State:
        # 应用动力学模型
        if "population" in self.dynamics_models:
            self.state = self.dynamics_models["population"].predict_next_state(
                self.state, actions or []
            )

        return super().step(actions)

    def get_population_summary(self) -> Dict[str, int]:
        """获取种群摘要"""
        summary = {}
        for entity in self.state.entities.values():
            if entity.active:
                summary[entity.name] = entity.get("population", 0)
        return summary

    def get_species_list(self) -> List[Dict[str, Any]]:
        """获取物种列表"""
        return [
            {
                "name": e.name,
                "type": "predator" if e.get("is_predator") else "prey",
                "population": e.get("population", 0),
                "energy": e.get("energy", 0),
                "active": e.active
            }
            for e in self.state.entities.values()
        ]


@register_world_model(
    name="simple_2_species",
    domain="ecology",
    description="简单的双物种种群动态模型（捕食者-猎物）",
    version="1.0.0",
    tags=["ecology", "simple", "lotka-volterra"]
)
class SimpleTwoSpeciesModel(EcologyWorldModel):
    """简化的双物种模型 - 经典Lotka-Volterra"""

    def initialize(self, config: Dict[str, Any]) -> None:
        default_config = {
            "species": [
                {"name": "predators", "type": "predator", "population": config.get("predators", 10)},
                {"name": "prey", "type": "prey", "population": config.get("prey", 50)}
            ],
            "environment": {"resource_availability": config.get("resources", 0.5)}
        }
        super().initialize(default_config)

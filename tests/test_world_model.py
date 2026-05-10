"""
Tests for the Universal World Model Framework
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.world_model.core import (
    WorldModel, Entity, EntityType, State, Action, Attribute,
    DynamicsModel, PerceptionModel, DecisionModel
)
from src.world_model.registry import (
    WorldModelRegistry, register_world_model, registry
)


class TestEntity:
    def test_entity_creation(self):
        entity = Entity(
            entity_id="test_1",
            entity_type=EntityType.AGENT,
            name="test_agent"
        )
        assert entity.entity_id == "test_1"
        assert entity.name == "test_agent"
        assert entity.active is True

    def test_entity_attributes(self):
        entity = Entity("1", EntityType.AGENT, "test")
        entity.set("energy", 100.0)
        entity.set("population", 50)

        assert entity.get("energy") == 100.0
        assert entity.get("population") == 50
        assert entity.get("nonexistent", "default") == "default"

    def test_entity_inactive(self):
        entity = Entity("1", EntityType.AGENT, "test")
        entity.active = False
        assert not entity.active


class TestState:
    def test_state_initializes(self):
        state = State()
        assert state.timestep == 0
        assert len(state.entities) == 0

    def test_state_add_entity(self):
        state = State()
        entity = Entity("1", EntityType.AGENT, "test")
        state.add_entity(entity)

        assert len(state.entities) == 1
        assert state.get_entity("1") is not None

    def test_state_remove_entity(self):
        state = State()
        entity = Entity("1", EntityType.AGENT, "test")
        state.add_entity(entity)
        state.remove_entity("1")

        # 移除只是标记为不活跃
        assert state.get_entity("1").active is False

    def test_get_entities_by_type(self):
        state = State()
        state.add_entity(Entity("1", EntityType.AGENT, "agent1"))
        state.add_entity(Entity("2", EntityType.RESOURCE, "resource1"))
        state.add_entity(Entity("3", EntityType.AGENT, "agent2"))

        agents = state.get_entities_by_type(EntityType.AGENT)
        assert len(agents) == 2

    def test_state_clone(self):
        state = State()
        state.add_entity(Entity("1", EntityType.AGENT, "test"))
        state.timestep = 5

        cloned = state.clone()
        assert cloned.timestep == 5
        assert len(cloned.entities) == 1
        assert cloned is not state


class TestSimpleModel(WorldModel):
    """测试用的简单世界模型"""

    def __init__(self):
        super().__init__("test_model", "test")

    def initialize(self, config):
        super().initialize(config)
        for i in range(config.get("num_entities", 3)):
            self.create_entity(
                name=f"entity_{i}",
                entity_type=EntityType.AGENT,
                attributes={"value": i}
            )

    def step(self, actions=None):
        # 简单的动力学：每个实体的 value 加 1
        for entity in self.state.entities.values():
            current = entity.get("value", 0)
            entity.set("value", current + 1)
        return super().step(actions)


class TestWorldModel:
    def test_model_initialization(self):
        model = TestSimpleModel()
        assert model.model_name == "test_model"
        assert model.domain == "test"
        assert not model.initialized

    def test_model_initialize(self):
        model = TestSimpleModel()
        model.initialize({"num_entities": 5})
        assert model.initialized
        assert len(model.state.entities) == 5

    def test_model_step(self):
        model = TestSimpleModel()
        model.initialize({"num_entities": 3})

        initial_values = [e.get("value") for e in model.state.entities.values()]

        model.step()

        final_values = [e.get("value") for e in model.state.entities.values()]
        assert all(f == i + 1 for f, i in zip(final_values, initial_values))

    def test_model_rollout(self):
        model = TestSimpleModel()
        model.initialize({"num_entities": 2})

        trajectory = model.rollout(10)
        assert len(trajectory) == 10
        assert model.state.timestep == 10

        # 检查值是否正确增加 (初始值为0,1, 每步+1, 10步后为10,11)
        values = sorted([e.get("value") for e in model.state.entities.values()])
        assert values[0] == 10  # entity_0: 0 + 10
        assert values[1] == 11  # entity_1: 1 + 10

    def test_get_statistics(self):
        model = TestSimpleModel()
        model.initialize({"num_entities": 5})
        model.step()

        stats = model.get_statistics()
        assert stats["model_name"] == "test_model"
        assert stats["total_entities"] == 5
        assert stats["active_entities"] == 5
        assert stats["timestep"] == 1


class TestDynamicsModel:
    class IncrementDynamics(DynamicsModel):
        def predict_next_state(self, current_state, actions):
            new_state = current_state.clone()
            for entity in new_state.entities.values():
                val = entity.get("value", 0)
                entity.set("value", val + 1)
            return new_state

    def test_dynamics_model(self):
        model = TestSimpleModel()
        model.initialize({"num_entities": 3})
        model.register_dynamics("increment", self.IncrementDynamics())

        # 手动应用动力学
        initial_values = [e.get("value") for e in model.state.entities.values()]
        model.state = model.dynamics_models["increment"].predict_next_state(
            model.state, []
        )
        final_values = [e.get("value") for e in model.state.entities.values()]

        assert all(f == i + 1 for f, i in zip(final_values, initial_values))


class TestRegistry:
    def test_registry_singleton(self):
        r1 = WorldModelRegistry()
        r2 = WorldModelRegistry()
        assert r1 is r2

    def test_register_and_list_models(self):
        @register_world_model(
            name="demo",
            domain="test",
            description="Test model"
        )
        class DemoModel(WorldModel):
            def initialize(self, config):
                super().initialize(config)

            def step(self, actions=None):
                return super().step(actions)

        models = registry.list_models()
        assert len(models) >= 1
        assert any(m["id"] == "test/demo" for m in models)

    def test_create_model_instance(self):
        @register_world_model(
            name="demo2",
            domain="test",
            description="Test model 2"
        )
        class Demo2Model(WorldModel):
            def initialize(self, config):
                super().initialize(config)
                self.create_entity("test", EntityType.AGENT, {"x": 1})

            def step(self, actions=None):
                return super().step(actions)

        instance = registry.create_instance("test/demo2", instance_id="test_instance")
        assert instance is not None
        assert instance.model_name == "demo2"

        # 从注册表获取实例
        retrieved = registry.get_instance("test_instance")
        assert retrieved is instance

    def test_list_and_destroy_instances(self):
        @register_world_model(
            name="demo3",
            domain="test",
            description="Test model 3"
        )
        class Demo3Model(WorldModel):
            def initialize(self, config):
                super().initialize(config)

            def step(self, actions=None):
                return super().step(actions)

        instance = registry.create_instance("test/demo3")
        instances = registry.list_instances()
        assert len(instances) >= 1

        # 销毁
        instance_id = instances[-1]["instance_id"]
        assert registry.destroy_instance(instance_id) is True
        assert registry.get_instance(instance_id) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

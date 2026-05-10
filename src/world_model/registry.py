"""
World Model Registry - 世界模型注册表

管理所有可用的领域世界模型，支持动态注册和发现
"""

from typing import Dict, List, Type, Any, Optional
from dataclasses import dataclass, field
import importlib
import inspect

from .core import WorldModel


@dataclass
class ModelMetadata:
    """世界模型元数据"""
    name: str
    domain: str
    description: str
    version: str
    author: str = ""
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)


class WorldModelRegistry:
    """世界模型注册表 - 单例模式"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._metadata = {}
            cls._instance._instances = {}
        return cls._instance

    def __init__(self):
        # 确保只初始化一次
        if hasattr(self, '_models'):
            return
        self._models: Dict[str, Type[WorldModel]] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._instances: Dict[str, WorldModel] = {}

    def register(
        self,
        model_class: Type[WorldModel],
        metadata: ModelMetadata
    ) -> None:
        """注册世界模型"""
        key = f"{metadata.domain}/{metadata.name}"
        self._models[key] = model_class
        self._metadata[key] = metadata

    def unregister(self, domain: str, name: str) -> None:
        """注销世界模型"""
        key = f"{domain}/{name}"
        if key in self._models:
            del self._models[key]
        if key in self._metadata:
            del self._metadata[key]

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用的世界模型"""
        return [
            {
                "id": key,
                "name": meta.name,
                "domain": meta.domain,
                "description": meta.description,
                "version": meta.version,
                "author": meta.author,
                "tags": meta.tags
            }
            for key, meta in self._metadata.items()
        ]

    def get_model_class(self, model_id: str) -> Optional[Type[WorldModel]]:
        """获取世界模型类"""
        return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        return self._metadata.get(model_id)

    def create_instance(
        self,
        model_id: str,
        instance_id: Optional[str] = None
    ) -> Optional[WorldModel]:
        """创建世界模型实例"""
        model_class = self._models.get(model_id)
        if not model_class:
            return None

        meta = self._metadata[model_id]
        instance = model_class(model_name=meta.name, domain=meta.domain)

        if instance_id:
            self._instances[instance_id] = instance
        else:
            import uuid
            instance_id = str(uuid.uuid4())[:12]
            self._instances[instance_id] = instance

        return instance

    def get_instance(self, instance_id: str) -> Optional[WorldModel]:
        """获取已创建的世界模型实例"""
        return self._instances.get(instance_id)

    def list_instances(self) -> List[Dict[str, Any]]:
        """列出所有运行中的模型实例"""
        return [
            {
                "instance_id": instance_id,
                "model_id": f"{instance.domain}/{instance.model_name}",
                "domain": instance.domain,
                "timestep": instance.state.timestep,
                "entity_count": len(instance.state.entities),
                "initialized": instance.initialized
            }
            for instance_id, instance in self._instances.items()
        ]

    def destroy_instance(self, instance_id: str) -> bool:
        """销毁模型实例"""
        if instance_id in self._instances:
            del self._instances[instance_id]
            return True
        return False


# 全局注册表实例
registry = WorldModelRegistry()


def register_world_model(
    name: str,
    domain: str,
    description: str = "",
    version: str = "1.0.0",
    author: str = "",
    tags: Optional[List[str]] = None
):
    """
    装饰器：注册世界模型

    Usage:
        @register_world_model(
            name="predator_prey",
            domain="ecology",
            description="捕食者-猎物生态系统模型",
            version="2.0.0",
            tags=["ecology", "population", "simulation"]
        )
        class PredatorPreyModel(WorldModel):
            ...
    """
    def decorator(cls: Type[WorldModel]) -> Type[WorldModel]:
        meta = ModelMetadata(
            name=name,
            domain=domain,
            description=description,
            version=version,
            author=author,
            tags=tags or []
        )
        registry.register(cls, meta)
        return cls
    return decorator


def load_models_from_module(module_path: str) -> int:
    """从模块自动加载所有世界模型"""
    try:
        module = importlib.import_module(module_path)
        loaded = 0

        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, WorldModel) and
                obj != WorldModel):

                # 检查是否有注册装饰器的元数据
                if hasattr(obj, '_world_model_meta'):
                    meta = obj._world_model_meta
                    registry.register(obj, meta)
                    loaded += 1

        return loaded
    except ImportError:
        return 0

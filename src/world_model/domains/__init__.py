"""
World Model Domains - 世界模型领域插件

已注册的领域模型：
- ecology/predator_prey: 生态系统捕食者-猎物模型
- ecology/simple_2_species: 简化双物种模型
- economy/market_simulation: 市场经济模拟
"""

from .ecology import EcologyWorldModel, SimpleTwoSpeciesModel
from .economy import EconomyWorldModel

__all__ = ["EcologyWorldModel", "SimpleTwoSpeciesModel", "EconomyWorldModel"]

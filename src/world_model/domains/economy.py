"""
Economy Domain World Model - 经济领域世界模型

演示通用世界模型如何模拟完全不同的领域
"""

import random
from typing import Dict, List, Any, Optional

from ..core import (
    WorldModel, EntityType, Action, State, DynamicsModel
)
from ..registry import register_world_model


class MarketDynamics(DynamicsModel):
    """市场动力学模型"""

    def predict_next_state(self, current_state: State, actions: List[Action]) -> State:
        new_state = current_state.clone()

        # 价格波动
        for entity in new_state.get_entities_by_type(EntityType.RESOURCE):
            if not entity.active:
                continue

            price = entity.get("price", 1.0)
            volatility = entity.get("volatility", 0.1)

            # 随机游走价格
            price_change = random.uniform(-volatility, volatility)
            new_price = price * (1 + price_change)
            entity.set("price", max(0.01, new_price))

        # 供需关系影响价格
        companies = new_state.get_entities_by_attribute("is_company", True)
        for company in companies:
            if not company.active:
                continue

            cash = company.get("cash", 1000.0)
            production = company.get("production", 10.0)
            efficiency = company.get("efficiency", 0.8)

            # 公司运营成本
            operating_cost = production * 2
            cash -= operating_cost

            # 收入
            revenue = production * random.uniform(0.8, 1.2)
            cash += revenue * efficiency

            company.set("cash", cash)

            # 破产检查
            if cash <= 0:
                company.active = False

        return new_state


@register_world_model(
    name="market_simulation",
    domain="economy",
    description="简单市场经济模拟，包含公司、商品和价格动态",
    version="1.0.0",
    tags=["economy", "market", "simulation", "companies"]
)
class EconomyWorldModel(WorldModel):
    """经济领域世界模型"""

    def __init__(self, model_name: str = "market_simulation", domain: str = "economy"):
        super().__init__(model_name, domain)
        self.register_dynamics("market", MarketDynamics())

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化经济系统

        Args:
            config: {
                "companies": [
                    {"name": "company_a", "cash": 10000, "production": 50},
                    ...
                ],
                "resources": [
                    {"name": "gold", "price": 100, "volatility": 0.05},
                    ...
                ]
            }
        """
        super().initialize(config)

        # 全局经济指标
        self.state.global_attributes.update({
            "gdp": config.get("initial_gdp", 100000),
            "inflation": config.get("inflation", 0.02),
            "interest_rate": config.get("interest_rate", 0.05)
        })

        # 创建公司
        for company_config in config.get("companies", []):
            attributes = {
                "cash": company_config.get("cash", 1000.0),
                "production": company_config.get("production", 10.0),
                "efficiency": company_config.get("efficiency", 0.8),
                "is_company": True
            }
            self.create_entity(
                name=company_config["name"],
                entity_type=EntityType.AGENT,
                attributes=attributes
            )

        # 创建资源/商品
        for resource_config in config.get("resources", []):
            attributes = {
                "price": resource_config.get("price", 1.0),
                "volatility": resource_config.get("volatility", 0.1),
                "total_supply": resource_config.get("supply", 1000),
                "is_resource": True
            }
            self.create_entity(
                name=resource_config["name"],
                entity_type=EntityType.RESOURCE,
                attributes=attributes
            )

    def step(self, actions: Optional[List[Action]] = None) -> State:
        if "market" in self.dynamics_models:
            self.state = self.dynamics_models["market"].predict_next_state(
                self.state, actions or []
            )

        # 更新GDP
        total_cash = sum(
            c.get("cash", 0) for c in self.state.get_entities_by_attribute("is_company", True)
        )
        self.state.global_attributes["gdp"] = total_cash * 2

        return super().step(actions)

    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场摘要"""
        companies = self.state.get_entities_by_attribute("is_company", True)
        resources = self.state.get_entities_by_attribute("is_resource", True)

        return {
            "gdp": self.state.global_attributes.get("gdp", 0),
            "inflation": self.state.global_attributes.get("inflation", 0),
            "active_companies": sum(1 for c in companies if c.active),
            "total_cash": sum(c.get("cash", 0) for c in companies if c.active),
            "resource_prices": {r.name: round(r.get("price", 0), 2) for r in resources if r.active}
        }

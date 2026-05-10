"""
Extended Ecosystem Simulator
扩展生态系统模拟器 - Phase 6

新增功能：
1. 植物种群动态（Plants Population Dynamics）
2. 季节更替系统（Seasonal Cycle System）
3. 地形与栖息地建模（Terrain & Habitat Modeling）
"""

import random
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


class Season(Enum):
    """季节类型"""
    SPRING = "春季"
    SUMMER = "夏季"
    AUTUMN = "秋季"
    WINTER = "冬季"


class TerrainType(Enum):
    """地形类型"""
    FOREST = "森林"
    GRASSLAND = "草原"
    WATER = "水域"
    MOUNTAIN = "山地"
    WETLAND = "湿地"
    DESERT = "荒漠"


class PlantType(Enum):
    """植物类型"""
    GRASS = "草"
    TREE = "树木"
    SHRUB = "灌木"
    FLOWER = "花卉"
    AQUATIC = "水生植物"


@dataclass
class TerrainCell:
    """地形单元格"""
    x: int
    y: int
    terrain_type: TerrainType
    fertility: float  # 0-1 土壤肥力
    water_availability: float  # 0-1 水资源可用性
    sunlight: float  # 0-1 阳光照射度


@dataclass
class PlantPopulation:
    """植物种群"""
    plant_type: PlantType
    name: str
    count: int
    growth_rate: float
    preferred_terrain: List[TerrainType]
    seasonal_effect: Dict[Season, float]  # 各季节的生长系数
    nutrition_value: float  # 营养价值（0-1）
    current_biomass: float  # 当前生物量


class SeasonalSystem:
    """季节系统"""

    SEASON_DURATION = 50  # 每季节持续的步数
    SEASON_ORDER = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]

    # 季节对生态的影响系数
    SEASON_EFFECTS = {
        Season.SPRING: {
            "plant_growth": 1.5,      # 植物生长加速
            "animal_reproduction": 1.3,  # 动物繁殖率提升
            "predator_activity": 1.0,    # 捕食者活跃度
            "prey_activity": 1.2,        # 猎物活跃度
            "resource_regen": 1.4,       # 资源再生率
        },
        Season.SUMMER: {
            "plant_growth": 1.8,
            "animal_reproduction": 1.5,
            "predator_activity": 1.3,
            "prey_activity": 1.4,
            "resource_regen": 1.2,
        },
        Season.AUTUMN: {
            "plant_growth": 0.8,
            "animal_reproduction": 0.7,
            "predator_activity": 1.2,
            "prey_activity": 0.9,
            "resource_regen": 0.6,
        },
        Season.WINTER: {
            "plant_growth": 0.2,
            "animal_reproduction": 0.1,
            "predator_activity": 0.6,
            "prey_activity": 0.5,
            "resource_regen": 0.3,
        },
    }

    def __init__(self):
        self.current_step = 0
        self.current_season_index = 0
        self.season_changes: List[Tuple[int, Season]] = []

    @property
    def current_season(self) -> Season:
        """获取当前季节"""
        return self.SEASON_ORDER[self.current_season_index]

    def step(self) -> Tuple[Season, bool]:
        """推进一个时间步，返回 (当前季节, 是否换季)"""
        self.current_step += 1
        season_changed = False

        # 检查是否换季
        if self.current_step % self.SEASON_DURATION == 0:
            self.current_season_index = (self.current_season_index + 1) % 4
            self.season_changes.append((self.current_step, self.current_season))
            season_changed = True

        return self.current_season, season_changed

    def get_effects(self) -> Dict[str, float]:
        """获取当前季节的生态影响系数"""
        return self.SEASON_EFFECTS[self.current_season]

    def get_days_remaining(self) -> int:
        """获取当前季节剩余步数"""
        return self.SEASON_DURATION - (self.current_step % self.SEASON_DURATION)


class TerrainMap:
    """地形图"""

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.cells: List[List[TerrainCell]] = []
        self._generate_terrain()

    def _generate_terrain(self):
        """生成随机地形图"""
        terrain_weights = {
            TerrainType.FOREST: 0.3,
            TerrainType.GRASSLAND: 0.25,
            TerrainType.WATER: 0.15,
            TerrainType.MOUNTAIN: 0.15,
            TerrainType.WETLAND: 0.1,
            TerrainType.DESERT: 0.05,
        }

        terrain_types = list(terrain_weights.keys())
        weights = list(terrain_weights.values())

        for y in range(self.height):
            row = []
            for x in range(self.width):
                terrain = random.choices(terrain_types, weights=weights, k=1)[0]

                # 根据地形设置属性
                if terrain == TerrainType.FOREST:
                    fertility = random.uniform(0.6, 0.9)
                    water = random.uniform(0.4, 0.7)
                    sunlight = random.uniform(0.3, 0.6)
                elif terrain == TerrainType.GRASSLAND:
                    fertility = random.uniform(0.5, 0.8)
                    water = random.uniform(0.3, 0.6)
                    sunlight = random.uniform(0.7, 1.0)
                elif terrain == TerrainType.WATER:
                    fertility = random.uniform(0.2, 0.5)
                    water = random.uniform(0.9, 1.0)
                    sunlight = random.uniform(0.5, 0.8)
                elif terrain == TerrainType.MOUNTAIN:
                    fertility = random.uniform(0.1, 0.4)
                    water = random.uniform(0.2, 0.5)
                    sunlight = random.uniform(0.8, 1.0)
                elif terrain == TerrainType.WETLAND:
                    fertility = random.uniform(0.7, 0.95)
                    water = random.uniform(0.8, 1.0)
                    sunlight = random.uniform(0.4, 0.7)
                else:  # DESERT
                    fertility = random.uniform(0.0, 0.2)
                    water = random.uniform(0.0, 0.1)
                    sunlight = random.uniform(0.9, 1.0)

                row.append(TerrainCell(x, y, terrain, fertility, water, sunlight))
            self.cells.append(row)

    def get_cell(self, x: int, y: int) -> Optional[TerrainCell]:
        """获取指定坐标的地形单元格"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[y][x]
        return None

    def get_terrain_distribution(self) -> Dict[str, int]:
        """获取地形分布统计"""
        distribution: Dict[str, int] = {}
        for row in self.cells:
            for cell in row:
                terrain_name = cell.terrain_type.value
                distribution[terrain_name] = distribution.get(terrain_name, 0) + 1
        return distribution

    def get_suitable_habitat(self, preferred_terrains: List[TerrainType]) -> List[Tuple[int, int]]:
        """获取适合特定物种的栖息地坐标"""
        suitable = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                if cell.terrain_type in preferred_terrains:
                    suitable.append((x, y))
        return suitable


class PlantEcosystem:
    """植物生态系统"""

    def __init__(self, terrain_map: TerrainMap):
        self.terrain_map = terrain_map
        self.plants: List[PlantPopulation] = []
        self._initialize_plants()

    def _initialize_plants(self):
        """初始化植物种群"""
        terrain_dist = self.terrain_map.get_terrain_distribution()
        total_cells = sum(terrain_dist.values())

        # 草
        grass_land = terrain_dist.get("草原", 0) + terrain_dist.get("湿地", 0)
        grass_count = int(grass_land * 10 / total_cells * 100) if total_cells > 0 else 50
        self.plants.append(PlantPopulation(
            plant_type=PlantType.GRASS,
            name="草本植物",
            count=max(20, grass_count),
            growth_rate=0.15,
            preferred_terrain=[TerrainType.GRASSLAND, TerrainType.WETLAND, TerrainType.FOREST],
            seasonal_effect={
                Season.SPRING: 1.5,
                Season.SUMMER: 2.0,
                Season.AUTUMN: 0.8,
                Season.WINTER: 0.1
            },
            nutrition_value=0.6,
            current_biomass=float(grass_count) * 0.5
        ))

        # 树木
        forest_cells = terrain_dist.get("森林", 0)
        tree_count = int(forest_cells * 5 / total_cells * 50) if total_cells > 0 else 30
        self.plants.append(PlantPopulation(
            plant_type=PlantType.TREE,
            name="乔木",
            count=max(10, tree_count),
            growth_rate=0.05,
            preferred_terrain=[TerrainType.FOREST, TerrainType.MOUNTAIN],
            seasonal_effect={
                Season.SPRING: 1.2,
                Season.SUMMER: 1.5,
                Season.AUTUMN: 0.5,
                Season.WINTER: 0.2
            },
            nutrition_value=0.3,
            current_biomass=float(tree_count) * 2.0
        ))

        # 灌木
        shrub_count = int((forest_cells + grass_land) * 3 / total_cells * 30) if total_cells > 0 else 20
        self.plants.append(PlantPopulation(
            plant_type=PlantType.SHRUB,
            name="灌木",
            count=max(10, shrub_count),
            growth_rate=0.08,
            preferred_terrain=[TerrainType.FOREST, TerrainType.GRASSLAND, TerrainType.WETLAND],
            seasonal_effect={
                Season.SPRING: 1.3,
                Season.SUMMER: 1.6,
                Season.AUTUMN: 0.6,
                Season.WINTER: 0.3
            },
            nutrition_value=0.4,
            current_biomass=float(shrub_count) * 0.8
        ))

        # 花卉
        self.plants.append(PlantPopulation(
            plant_type=PlantType.FLOWER,
            name="开花植物",
            count=25,
            growth_rate=0.12,
            preferred_terrain=[TerrainType.GRASSLAND, TerrainType.FOREST, TerrainType.WETLAND],
            seasonal_effect={
                Season.SPRING: 2.0,
                Season.SUMMER: 1.5,
                Season.AUTUMN: 0.3,
                Season.WINTER: 0.0
            },
            nutrition_value=0.2,
            current_biomass=12.5
        ))

        # 水生植物
        water_cells = terrain_dist.get("水域", 0) + terrain_dist.get("湿地", 0)
        aquatic_count = int(water_cells * 8 / total_cells * 40) if total_cells > 0 else 15
        self.plants.append(PlantPopulation(
            plant_type=PlantType.AQUATIC,
            name="水生植物",
            count=max(5, aquatic_count),
            growth_rate=0.1,
            preferred_terrain=[TerrainType.WATER, TerrainType.WETLAND],
            seasonal_effect={
                Season.SPRING: 1.4,
                Season.SUMMER: 1.8,
                Season.AUTUMN: 0.7,
                Season.WINTER: 0.2
            },
            nutrition_value=0.5,
            current_biomass=float(aquatic_count) * 0.6
        ))

    def step(self, season: Season, season_effects: Dict[str, float]) -> Dict[str, Any]:
        """推进一个时间步"""
        changes = {}

        for plant in self.plants:
            # 计算生长率
            base_growth = plant.growth_rate
            season_factor = plant.seasonal_effect.get(season, 1.0)
            global_season_factor = season_effects.get("plant_growth", 1.0)

            # 计算适宜栖息地的影响
            suitable_cells = self.terrain_map.get_suitable_habitat(plant.preferred_terrain)
            habitat_factor = min(1.0, len(suitable_cells) / 100)

            # 总生长率
            total_growth_rate = base_growth * season_factor * global_season_factor * (0.5 + habitat_factor * 0.5)

            # 计算种群变化（逻辑斯谛增长模型）
            carrying_capacity = len(suitable_cells) * 5  # 承载能力
            if carrying_capacity > 0:
                growth = total_growth_rate * plant.count * (1 - plant.count / carrying_capacity)
            else:
                growth = -0.05 * plant.count  # 不适宜栖息地导致衰退

            # 更新种群数量和生物量
            old_count = plant.count
            plant.count = max(1, int(plant.count + growth))
            plant.current_biomass = plant.count * (0.5 + plant.nutrition_value)

            changes[plant.name] = {
                "old_count": old_count,
                "new_count": plant.count,
                "growth": growth,
                "biomass": plant.current_biomass
            }

        return changes

    def get_total_biomass(self) -> float:
        """获取总植物生物量"""
        return sum(p.current_biomass for p in self.plants)

    def get_plant_statistics(self) -> Dict[str, Any]:
        """获取植物种群统计"""
        stats = {
            "total_species": len(self.plants),
            "total_plants": sum(p.count for p in self.plants),
            "total_biomass": self.get_total_biomass(),
            "species": {}
        }

        for plant in self.plants:
            stats["species"][plant.name] = {
                "count": plant.count,
                "biomass": plant.current_biomass,
                "growth_rate": plant.growth_rate,
                "nutrition_value": plant.nutrition_value
            }

        return stats

    def consume_plant_biomass(self, amount: float, herbivore_type: str) -> float:
        """植食动物消耗植物生物量，返回实际消耗量"""
        consumed = 0.0
        remaining = amount

        # 优先消耗营养价值高的植物
        sorted_plants = sorted(self.plants, key=lambda p: -p.nutrition_value)

        for plant in sorted_plants:
            if remaining <= 0:
                break
            if plant.current_biomass > 0:
                consume_from_this = min(remaining, plant.current_biomass * 0.1)  # 最多消耗10%
                plant.current_biomass -= consume_from_this
                consumed += consume_from_this
                remaining -= consume_from_this

        return consumed


class ExtendedEcosystemSimulator:
    """扩展生态系统模拟器"""

    def __init__(
        self,
        map_width: int = 20,
        map_height: int = 20,
        enable_seasons: bool = True,
        enable_plants: bool = True,
        enable_terrain: bool = True
    ):
        self.enable_seasons = enable_seasons
        self.enable_plants = enable_plants
        self.enable_terrain = enable_terrain

        # 初始化子系统
        if enable_seasons:
            self.season_system = SeasonalSystem()
        else:
            self.season_system = None

        if enable_terrain:
            self.terrain_map = TerrainMap(map_width, map_height)
        else:
            self.terrain_map = None

        if enable_plants and self.terrain_map:
            self.plant_ecosystem = PlantEcosystem(self.terrain_map)
        else:
            self.plant_ecosystem = None

        self.step_count = 0
        self.history: List[Dict[str, Any]] = []

    def step(self) -> Dict[str, Any]:
        """推进一个时间步"""
        self.step_count += 1
        step_data: Dict[str, Any] = {"step": self.step_count}

        # 季节系统
        if self.season_system:
            season, changed = self.season_system.step()
            step_data["season"] = season.value
            step_data["season_changed"] = changed
            step_data["season_effects"] = self.season_system.get_effects()
            step_data["days_remaining_in_season"] = self.season_system.get_days_remaining()
        else:
            step_data["season"] = None
            step_data["season_changed"] = False
            step_data["season_effects"] = {"plant_growth": 1.0}

        # 植物生态系统
        if self.plant_ecosystem:
            plant_changes = self.plant_ecosystem.step(
                self.season_system.current_season if self.season_system else Season.SPRING,
                step_data["season_effects"]
            )
            step_data["plant_changes"] = plant_changes
            step_data["plant_stats"] = self.plant_ecosystem.get_plant_statistics()

        self.history.append(step_data)
        return step_data

    def run(self, max_steps: int = 500) -> List[Dict[str, Any]]:
        """运行模拟"""
        for _ in range(max_steps):
            self.step()
        return self.history

    def get_summary(self) -> Dict[str, Any]:
        """获取模拟摘要"""
        summary = {
            "total_steps": self.step_count,
            "features_enabled": {
                "seasons": self.enable_seasons,
                "plants": self.enable_plants,
                "terrain": self.enable_terrain
            }
        }

        if self.season_system:
            summary["current_season"] = self.season_system.current_season.value
            summary["season_changes"] = len(self.season_system.season_changes)

        if self.terrain_map:
            summary["terrain_distribution"] = self.terrain_map.get_terrain_distribution()

        if self.plant_ecosystem:
            summary["plant_statistics"] = self.plant_ecosystem.get_plant_statistics()

        return summary

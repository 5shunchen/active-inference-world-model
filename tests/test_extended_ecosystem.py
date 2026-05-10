"""
Tests for Extended Ecosystem Simulator (Phase 6)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.extended_ecosystem import (
    Season,
    TerrainType,
    PlantType,
    SeasonalSystem,
    TerrainMap,
    PlantPopulation,
    PlantEcosystem,
    ExtendedEcosystemSimulator
)


class TestSeasonEnum:
    def test_season_values(self):
        assert Season.SPRING.value == "春季"
        assert Season.SUMMER.value == "夏季"
        assert Season.AUTUMN.value == "秋季"
        assert Season.WINTER.value == "冬季"


class TestSeasonalSystem:
    def test_season_system_initializes(self):
        system = SeasonalSystem()
        assert system.current_season == Season.SPRING
        assert system.current_step == 0

    def test_season_progresses(self):
        system = SeasonalSystem()

        # 运行到接近换季
        for _ in range(SeasonalSystem.SEASON_DURATION - 1):
            season, changed = system.step()
            assert season == Season.SPRING
            assert not changed

        # 换季
        season, changed = system.step()
        assert season == Season.SUMMER
        assert changed

    def test_full_season_cycle(self):
        system = SeasonalSystem()
        seasons_seen = set()

        for _ in range(SeasonalSystem.SEASON_DURATION * 4):
            season, _ = system.step()
            seasons_seen.add(season)

        assert len(seasons_seen) == 4
        assert system.current_season == Season.SPRING  # 循环回到春季

    def test_season_effects_exist(self):
        system = SeasonalSystem()
        effects = system.get_effects()
        assert "plant_growth" in effects
        assert "animal_reproduction" in effects
        assert "predator_activity" in effects
        assert "prey_activity" in effects
        assert "resource_regen" in effects
        assert effects["plant_growth"] > 0

    def test_days_remaining(self):
        system = SeasonalSystem()
        assert system.get_days_remaining() == SeasonalSystem.SEASON_DURATION

        system.step()
        assert system.get_days_remaining() == SeasonalSystem.SEASON_DURATION - 1


class TestTerrainType:
    def test_terrain_types(self):
        assert TerrainType.FOREST.value == "森林"
        assert TerrainType.GRASSLAND.value == "草原"
        assert TerrainType.WATER.value == "水域"
        assert TerrainType.MOUNTAIN.value == "山地"
        assert TerrainType.WETLAND.value == "湿地"
        assert TerrainType.DESERT.value == "荒漠"


class TestTerrainMap:
    def test_terrain_map_initializes(self):
        terrain = TerrainMap(width=10, height=10)
        assert terrain.width == 10
        assert terrain.height == 10
        assert len(terrain.cells) == 10
        assert len(terrain.cells[0]) == 10

    def test_get_cell_valid(self):
        terrain = TerrainMap(width=10, height=10)
        cell = terrain.get_cell(5, 5)
        assert cell is not None
        assert cell.x == 5
        assert cell.y == 5

    def test_get_cell_invalid(self):
        terrain = TerrainMap(width=10, height=10)
        assert terrain.get_cell(-1, 5) is None
        assert terrain.get_cell(10, 5) is None
        assert terrain.get_cell(5, -1) is None
        assert terrain.get_cell(5, 10) is None

    def test_terrain_distribution(self):
        terrain = TerrainMap(width=10, height=10)
        distribution = terrain.get_terrain_distribution()
        assert len(distribution) > 0
        assert sum(distribution.values()) == 100  # 10x10 map

    def test_cell_properties(self):
        terrain = TerrainMap(width=5, height=5)
        cell = terrain.get_cell(0, 0)
        assert 0 <= cell.fertility <= 1
        assert 0 <= cell.water_availability <= 1
        assert 0 <= cell.sunlight <= 1

    def test_suitable_habitat(self):
        terrain = TerrainMap(width=10, height=10)
        suitable = terrain.get_suitable_habitat([TerrainType.FOREST])
        assert isinstance(suitable, list)


class TestPlantType:
    def test_plant_types(self):
        assert PlantType.GRASS.value == "草"
        assert PlantType.TREE.value == "树木"
        assert PlantType.SHRUB.value == "灌木"
        assert PlantType.FLOWER.value == "花卉"
        assert PlantType.AQUATIC.value == "水生植物"


class TestPlantPopulation:
    def test_plant_population_initializes(self):
        plant = PlantPopulation(
            plant_type=PlantType.GRASS,
            name="测试草",
            count=100,
            growth_rate=0.1,
            preferred_terrain=[TerrainType.GRASSLAND],
            seasonal_effect={Season.SPRING: 1.5, Season.SUMMER: 2.0},
            nutrition_value=0.6,
            current_biomass=50.0
        )
        assert plant.count == 100
        assert plant.growth_rate == 0.1
        assert plant.nutrition_value == 0.6


class TestPlantEcosystem:
    def test_plant_ecosystem_initializes(self):
        terrain = TerrainMap(width=10, height=10)
        plant_system = PlantEcosystem(terrain)
        assert len(plant_system.plants) == 5  # 5种植物

    def test_plant_names_correct(self):
        terrain = TerrainMap(width=10, height=10)
        plant_system = PlantEcosystem(terrain)
        plant_names = [p.name for p in plant_system.plants]
        assert "草本植物" in plant_names
        assert "乔木" in plant_names
        assert "灌木" in plant_names
        assert "开花植物" in plant_names
        assert "水生植物" in plant_names

    def test_plant_growth_step(self):
        terrain = TerrainMap(width=10, height=10)
        plant_system = PlantEcosystem(terrain)
        initial_counts = [p.count for p in plant_system.plants]

        # 春季生长
        changes = plant_system.step(
            Season.SPRING,
            {"plant_growth": 1.5}
        )

        assert len(changes) == 5
        # 至少有一种植物数量发生了变化
        final_counts = [p.count for p in plant_system.plants]
        assert any(i != f for i, f in zip(initial_counts, final_counts))

    def test_get_total_biomass(self):
        terrain = TerrainMap(width=10, height=10)
        plant_system = PlantEcosystem(terrain)
        biomass = plant_system.get_total_biomass()
        assert biomass > 0

    def test_get_plant_statistics(self):
        terrain = TerrainMap(width=10, height=10)
        plant_system = PlantEcosystem(terrain)
        stats = plant_system.get_plant_statistics()

        assert "total_species" in stats
        assert "total_plants" in stats
        assert "total_biomass" in stats
        assert "species" in stats
        assert len(stats["species"]) == 5

    def test_consume_plant_biomass(self):
        terrain = TerrainMap(width=10, height=10)
        plant_system = PlantEcosystem(terrain)
        initial_biomass = plant_system.get_total_biomass()

        consumed = plant_system.consume_plant_biomass(10.0, "deer")
        final_biomass = plant_system.get_total_biomass()

        assert consumed > 0
        assert final_biomass < initial_biomass


class TestExtendedEcosystemSimulator:
    def test_simulator_initializes_default(self):
        sim = ExtendedEcosystemSimulator()
        assert sim.enable_seasons is True
        assert sim.enable_plants is True
        assert sim.enable_terrain is True
        assert sim.season_system is not None
        assert sim.terrain_map is not None
        assert sim.plant_ecosystem is not None

    def test_simulator_disabled_features(self):
        sim = ExtendedEcosystemSimulator(
            enable_seasons=False,
            enable_plants=False,
            enable_terrain=False
        )
        assert sim.season_system is None
        assert sim.terrain_map is None
        assert sim.plant_ecosystem is None

    def test_single_step(self):
        sim = ExtendedEcosystemSimulator()
        data = sim.step()

        assert "step" in data
        assert data["step"] == 1
        assert "season" in data
        assert "season_changed" in data
        assert "season_effects" in data
        assert "plant_changes" in data
        assert "plant_stats" in data

    def test_multiple_steps(self):
        sim = ExtendedEcosystemSimulator()
        history = sim.run(max_steps=10)

        assert len(history) == 10
        assert sim.step_count == 10
        assert history[-1]["step"] == 10

    def test_season_changes_during_simulation(self):
        sim = ExtendedEcosystemSimulator()
        steps_per_season = SeasonalSystem.SEASON_DURATION

        # 运行超过一个季节的时间
        for _ in range(steps_per_season + 5):
            sim.step()

        # 应该有一次换季
        assert sim.season_system.current_season == Season.SUMMER
        assert len(sim.season_system.season_changes) == 1

    def test_get_summary(self):
        sim = ExtendedEcosystemSimulator()
        sim.run(max_steps=50)
        summary = sim.get_summary()

        assert "total_steps" in summary
        assert "features_enabled" in summary
        assert "current_season" in summary
        assert "terrain_distribution" in summary
        assert "plant_statistics" in summary
        assert summary["total_steps"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Stage 2: Language Parser
Maps natural language prompts to structured scene configurations
"""

import json
import os
from typing import Dict, Any, Optional
import re


class LanguageParser:
    """
    Parses natural language prompts and generates structured scene configurations.
    Uses keyword matching and template lookup for scene configuration.
    """

    def __init__(self, templates_path: Optional[str] = None):
        if templates_path is None:
            templates_path = os.path.join(
                os.path.dirname(__file__), "scene_templates.json"
            )

        with open(templates_path, "r", encoding="utf-8") as f:
            self.templates = json.load(f)

        self.agent_keywords = {
            "tiger": ["老虎", "tiger", "虎"],
            "deer": ["鹿", "deer"],
        }

        self.biome_keywords = {
            "jungle": ["丛林", "jungle", "雨林", "tropical"],
            "savanna": ["草原", "savanna", "savannah", "savanna"],
            "desert": ["沙漠", "desert"],
        }

    def parse(self, prompt: str) -> Dict[str, Any]:
        """
        Parse a natural language prompt and return scene configuration.

        Args:
            prompt: Natural language prompt like "给我一个老虎的生命历程"

        Returns:
            Structured scene configuration dictionary
        """
        # First try exact template match
        for template_name, config in self.templates["prompt_templates"].items():
            if template_name in prompt:
                return config.copy()

        # If no exact match, do keyword-based parsing
        return self._keyword_parse(prompt)

    def _keyword_parse(self, prompt: str) -> Dict[str, Any]:
        """Parse prompt using keyword matching"""
        config = {
            "biome": "jungle",
            "agent_type": "tiger",
            "life_stage": "cub",
            "duration": "lifespan",
            "entities": ["prey", "water_source", "trees"],
            "physics": "default_ecology"
        }

        # Detect agent type
        for agent_type, keywords in self.agent_keywords.items():
            for kw in keywords:
                if kw in prompt.lower():
                    config["agent_type"] = agent_type
                    break

        # Detect biome
        for biome_type, keywords in self.biome_keywords.items():
            for kw in keywords:
                if kw in prompt.lower():
                    config["biome"] = biome_type
                    break

        # Detect life stage
        if any(w in prompt.lower() for w in ["成年", "adult", "grown"]):
            config["life_stage"] = "adult"
        elif any(w in prompt.lower() for w in ["老", "老年", "elder", "old"]):
            config["life_stage"] = "elderly"
        elif any(w in prompt.lower() for w in ["幼年", "幼崽", "baby", "young", "cub"]):
            config["life_stage"] = "cub"

        # Detect duration
        if any(w in prompt.lower() for w in ["一生", "生命历程", "lifespan", "whole life", "entire life"]):
            config["duration"] = "lifespan"

        return config

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        return self.templates["agents"].get(agent_type, {})

    def get_biome_config(self, biome: str) -> Dict[str, Any]:
        """Get biome-specific configuration"""
        return self.templates["biomes"].get(biome, {})

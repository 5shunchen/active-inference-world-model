"""
Tests for LLM-powered Enhanced Narrative Generator
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_narrative import (
    EventType,
    EcosystemEvent,
    EcosystemAnalyzer,
    ChineseNarrativeGenerator,
    generate_ecosystem_report_from_steps,
    generate_simple_narrative,
    LLMEnhancedNarrator,
    LLMConfig,
    LLMProvider,
    generate_llm_enhanced_report
)


class TestEventType:
    def test_event_types_exist(self):
        assert EventType.BIRTH.value == "出生"
        assert EventType.DEATH.value == "死亡"
        assert EventType.PREDATION.value == "捕食"
        assert EventType.POPULATION_RISE.value == "种群增长"
        assert EventType.POPULATION_FALL.value == "种群下降"
        assert EventType.EXTINCTION.value == "灭绝"
        assert EventType.BALANCE.value == "生态平衡"


class TestEcosystemAnalyzer:
    def test_analyzer_initializes(self):
        steps_data = [
            {"tigers": 1, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 10}
        ]
        analyzer = EcosystemAnalyzer(steps_data)
        assert len(analyzer.steps) == 1

    def test_detect_population_rise(self):
        steps_data = []
        # Steady population, then rapid rise
        for i in range(10):
            steps_data.append({"tigers": 1, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 10})
        # Rapid rise in rabbits
        for i in range(10):
            steps_data.append({"tigers": 1, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 20 + i})

        analyzer = EcosystemAnalyzer(steps_data)
        events = analyzer.detect_all_events()

        rise_events = [e for e in events if e.event_type == EventType.POPULATION_RISE]
        assert len(rise_events) > 0
        assert rise_events[0].species == "rabbits"

    def test_detect_population_fall(self):
        steps_data = []
        for i in range(10):
            steps_data.append({"tigers": 1, "wolves": 1, "deer": 20, "foxes": 2, "rabbits": 10})
        # Rapid fall in deer
        for i in range(10):
            steps_data.append({"tigers": 1, "wolves": 1, "deer": max(0, 20 - i * 2), "foxes": 2, "rabbits": 10})

        analyzer = EcosystemAnalyzer(steps_data)
        events = analyzer.detect_all_events()

        fall_events = [e for e in events if e.event_type == EventType.POPULATION_FALL]
        assert len(fall_events) > 0
        assert fall_events[0].species == "deer"

    def test_detect_extinction(self):
        steps_data = []
        # Tigers alive first, then go extinct
        for i in range(5):
            steps_data.append({"tigers": 1, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 10})
        for i in range(5):
            steps_data.append({"tigers": 0, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 10})

        analyzer = EcosystemAnalyzer(steps_data)
        events = analyzer.detect_all_events()

        extinction_events = [e for e in events if e.event_type == EventType.EXTINCTION]
        assert len(extinction_events) == 1
        assert extinction_events[0].species == "tigers"

    def test_no_extinction_if_never_alive(self):
        steps_data = []
        for i in range(10):
            steps_data.append({"tigers": 0, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 10})

        analyzer = EcosystemAnalyzer(steps_data)
        events = analyzer.detect_all_events()

        extinction_events = [e for e in events if e.event_type == EventType.EXTINCTION]
        assert len(extinction_events) == 0

    def test_get_summary_statistics(self):
        steps_data = [
            {"tigers": 2, "wolves": 2, "deer": 10, "foxes": 3, "rabbits": 15},
            {"tigers": 1, "wolves": 2, "deer": 8, "foxes": 3, "rabbits": 20},
        ]
        analyzer = EcosystemAnalyzer(steps_data)
        stats = analyzer.get_summary_statistics()

        assert stats["total_steps"] == 2
        assert stats["initial_population"]["tigers"] == 2
        assert stats["final_population"]["tigers"] == 1
        assert stats["population_change"]["tigers"] == -50.0


class TestChineseNarrativeGenerator:
    def test_generator_initializes(self):
        steps_data = [
            {"tigers": 1, "wolves": 1, "deer": 5, "foxes": 2, "rabbits": 10}
        ]
        analyzer = EcosystemAnalyzer(steps_data)
        analyzer.detect_all_events()
        stats = analyzer.get_summary_statistics()

        generator = ChineseNarrativeGenerator(analyzer.events, stats)
        assert generator.stats is not None

    def test_translate_species(self):
        steps_data = []
        analyzer = EcosystemAnalyzer(steps_data)
        stats = analyzer.get_summary_statistics()

        generator = ChineseNarrativeGenerator([], stats)
        assert generator._translate_species("tigers") == "老虎"
        assert generator._translate_species("wolves") == "灰狼"
        assert generator._translate_species("deer") == "梅花鹿"
        assert generator._translate_species("foxes") == "狐狸"
        assert generator._translate_species("rabbits") == "野兔"

    def test_generate_executive_summary(self):
        steps_data = [
            {"tigers": 2, "wolves": 2, "deer": 10, "foxes": 3, "rabbits": 15},
            {"tigers": 1, "wolves": 2, "deer": 8, "foxes": 3, "rabbits": 20},
        ]
        analyzer = EcosystemAnalyzer(steps_data)
        analyzer.detect_all_events()
        stats = analyzer.get_summary_statistics()

        generator = ChineseNarrativeGenerator(analyzer.events, stats)
        summary = generator.generate_executive_summary()

        assert "生态系统模拟报告" in summary
        assert "总模拟步数" in summary
        assert "存活物种" in summary
        assert "种群变化" in summary

    def test_generate_timeline(self):
        steps_data = []
        for i in range(20):
            steps_data.append({
                "tigers": max(0, 5 - i // 5),
                "wolves": 2,
                "deer": 10 + i,
                "foxes": 2,
                "rabbits": 10
            })

        analyzer = EcosystemAnalyzer(steps_data)
        events = analyzer.detect_all_events()
        stats = analyzer.get_summary_statistics()

        generator = ChineseNarrativeGenerator(events, stats)
        timeline = generator.generate_timeline()

        assert isinstance(timeline, str)
        assert len(timeline) > 0

    def test_generate_insights(self):
        steps_data = []
        for i in range(10):
            steps_data.append({
                "tigers": 1,
                "wolves": 1,
                "deer": 5,
                "foxes": 2,
                "rabbits": 10
            })

        analyzer = EcosystemAnalyzer(steps_data)
        events = analyzer.detect_all_events()
        stats = analyzer.get_summary_statistics()

        generator = ChineseNarrativeGenerator(events, stats)
        insights = generator.generate_insights()

        assert "生态洞察" in insights
        assert "生物多样性" in insights

    def test_generate_full_report(self):
        steps_data = []
        for i in range(30):
            steps_data.append({
                "tigers": max(0, 2 - i // 15),
                "wolves": 2,
                "deer": 10 + i,
                "foxes": 2,
                "rabbits": 10 + i * 2
            })

        report = generate_ecosystem_report_from_steps(steps_data)

        assert "生态系统模拟报告" in report
        assert "关键生态事件时间线" in report
        assert "生态洞察" in report


class TestSimpleNarrativeAPI:
    def test_generate_simple_narrative(self):
        steps_data = []
        for i in range(20):
            steps_data.append({
                "tigers": max(0, 2 - i // 10),
                "wolves": 2,
                "deer": 10 + i,
                "foxes": 2,
                "rabbits": 10 + i * 2
            })

        result = generate_simple_narrative(steps_data)

        assert "events" in result
        assert "statistics" in result
        assert isinstance(result["events"], list)
        assert isinstance(result["statistics"], dict)

        if result["events"]:
            assert "step" in result["events"][0]
            assert "type" in result["events"][0]
            assert "species" in result["events"][0]
            assert "description" in result["events"][0]
            assert "importance" in result["events"][0]


class TestLLMProvider:
    def test_llm_provider_enum(self):
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.DUMMY.value == "dummy"


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == LLMProvider.DUMMY
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_custom_config(self):
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test_key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=500
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key == "test_key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 500


class TestLLMEnhancedNarrator:
    def test_narrator_initializes(self):
        config = LLMConfig(provider=LLMProvider.DUMMY)
        narrator = LLMEnhancedNarrator(config)
        assert narrator is not None
        assert narrator.config.provider == LLMProvider.DUMMY

    def test_build_prompt(self):
        narrator = LLMEnhancedNarrator()
        events = [
            EcosystemEvent(
                step=10,
                event_type=EventType.POPULATION_RISE,
                species="tigers",
                description="老虎种群增长",
                magnitude=0.8
            )
        ]
        stats = {"total_steps": 100, "surviving_species": ["tigers"]}

        prompt = narrator._build_prompt(events, stats)
        assert "老虎" in prompt
        assert "生态系统" in prompt

    def test_fallback_narrative(self):
        narrator = LLMEnhancedNarrator()
        narrative = narrator._fallback_narrative("test prompt")
        assert len(narrative) > 0
        assert "生态系统" in narrative

    def test_generate_enhanced_narrative_dummy(self):
        narrator = LLMEnhancedNarrator()
        events = [
            EcosystemEvent(
                step=10,
                event_type=EventType.POPULATION_RISE,
                species="tigers",
                description="老虎种群增长",
                magnitude=0.8
            )
        ]
        stats = {"total_steps": 100, "surviving_species": ["tigers"]}

        narrative = narrator.generate_enhanced_narrative(events, stats)
        assert len(narrative) > 0
        assert "生态系统" in narrative


class TestLLMEnhancedReport:
    def test_generate_llm_enhanced_report_dummy(self):
        steps_data = []
        for i in range(20):
            steps_data.append({
                "tigers": max(0, 2 - i // 15),
                "wolves": 2,
                "deer": 10 + i,
                "foxes": 2,
                "rabbits": 10 + i * 2
            })

        result = generate_llm_enhanced_report(steps_data, provider="dummy")

        assert "statistics" in result
        assert "events" in result
        assert "base_report" in result
        assert "enhanced_narrative" in result
        assert "llm_provider" in result
        assert result["llm_provider"] == "dummy"
        assert len(result["enhanced_narrative"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

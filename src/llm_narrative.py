"""
LLM-powered Enhanced Narrative Generator
智能叙事生成器 - 用中文自然语言描述生态系统发生的事件

支持多种LLM后端：OpenAI、Claude、本地模型
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """生态系统事件类型"""
    BIRTH = "出生"
    DEATH = "死亡"
    PREDATION = "捕食"
    POPULATION_RISE = "种群增长"
    POPULATION_FALL = "种群下降"
    EXTINCTION = "灭绝"
    DOMINANCE = "优势地位"
    RESOURCE_CRISIS = "资源危机"
    BALANCE = "生态平衡"
    COMPETITION = "种间竞争"


@dataclass
class EcosystemEvent:
    """生态系统事件"""
    step: int
    event_type: EventType
    species: str
    description: str
    magnitude: float  # 0-1 事件重要程度


class EcosystemAnalyzer:
    """生态系统数据分析器"""

    def __init__(self, steps_data: List[Dict[str, Any]]):
        self.steps = steps_data
        self.population_history = self._extract_population_history()
        self.events: List[EcosystemEvent] = []

    def _extract_population_history(self) -> Dict[str, List[int]]:
        """提取种群历史数据"""
        history = {
            "tigers": [],
            "wolves": [],
            "deer": [],
            "foxes": [],
            "rabbits": [],
        }

        for step in self.steps:
            history["tigers"].append(step.get("tigers", 0))
            history["wolves"].append(step.get("wolves", 0))
            history["deer"].append(step.get("deer", 0))
            history["foxes"].append(step.get("foxes", 0))
            history["rabbits"].append(step.get("rabbits", 0))

        return history

    def detect_all_events(self) -> List[EcosystemEvent]:
        """检测所有重要生态事件"""
        self.events = []

        # 检测种群急剧变化
        self._detect_population_changes()

        # 检测灭绝事件
        self._detect_extinctions()

        # 检测捕食者-猎物模式
        self._detect_predator_prey_patterns()

        # 检测生态平衡/失衡
        self._detect_ecosystem_balance()

        # 按重要性排序
        self.events.sort(key=lambda e: (-e.magnitude, e.step))
        return self.events

    def _detect_population_changes(self):
        """检测种群急剧增长或下降"""
        window_size = max(5, len(self.steps) // 10)

        for species, counts in self.population_history.items():
            if len(counts) < window_size * 2:
                continue

            for i in range(window_size, len(counts)):
                prev_avg = sum(counts[i - window_size:i]) / window_size
                current = counts[i]

                if prev_avg > 0:
                    change_ratio = current / prev_avg

                    # 种群急剧增长 (>50%)
                    if change_ratio > 1.5 and current > 2:
                        self.events.append(EcosystemEvent(
                            step=i,
                            event_type=EventType.POPULATION_RISE,
                            species=species,
                            description=f"{species}种群数量显著增长，达到{current}只",
                            magnitude=min(1.0, (change_ratio - 1) * 2)
                        ))

                    # 种群急剧下降 (>40%)
                    elif change_ratio < 0.6 and current > 0:
                        self.events.append(EcosystemEvent(
                            step=i,
                            event_type=EventType.POPULATION_FALL,
                            species=species,
                            description=f"{species}种群数量急剧下降至{current}只",
                            magnitude=min(1.0, (1 - change_ratio) * 2)
                        ))

    def _detect_extinctions(self):
        """检测物种灭绝"""
        for species, counts in self.population_history.items():
            was_alive = False
            for i, count in enumerate(counts):
                if count > 0:
                    was_alive = True
                elif was_alive and count == 0:
                    self.events.append(EcosystemEvent(
                        step=i,
                        event_type=EventType.EXTINCTION,
                        species=species,
                        description=f"{species}已灭绝！生态系统永远失去了这个物种",
                        magnitude=1.0
                    ))
                    break

    def _detect_predator_prey_patterns(self):
        """检测捕食者-猎物动态模式"""
        # 老虎 vs 鹿
        tigers = self.population_history["tigers"]
        deer = self.population_history["deer"]
        rabbits = self.population_history["rabbits"]

        for i in range(1, min(len(tigers), len(deer), len(rabbits))):
            # 捕食者增加后猎物减少
            if tigers[i] > tigers[i-1] and deer[i] < deer[i-1] * 0.8 and deer[i-1] > 5:
                self.events.append(EcosystemEvent(
                    step=i,
                    event_type=EventType.PREDATION,
                    species="tigers",
                    description=f"老虎种群增加后，鹿种群明显下降，从{deer[i-1]}只减少到{deer[i]}只",
                    magnitude=0.7
                ))

    def _detect_ecosystem_balance(self):
        """检测生态系统整体平衡状态"""
        if len(self.steps) < 50:
            return

        # 检查最后20步的稳定性
        final_steps = self.steps[-20:]

        total_alive = [
            step.get("tigers", 0) + step.get("wolves", 0) + step.get("deer", 0) +
            step.get("foxes", 0) + step.get("rabbits", 0)
            for step in final_steps
        ]

        # 检查是否有多个物种共存
        final = self.steps[-1]
        surviving = sum(1 for s in ["tigers", "wolves", "deer", "foxes", "rabbits"] if final.get(s, 0) > 0)

        if surviving >= 4 and max(total_alive) / (min(total_alive) + 1) < 1.3:
            self.events.append(EcosystemEvent(
                step=len(self.steps),
                event_type=EventType.BALANCE,
                species="ecosystem",
                description=f"生态系统达到动态平衡！{surviving}个物种稳定共存，总种群约{int(sum(total_alive)/len(total_alive))}只",
                magnitude=0.9
            ))

    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if not self.steps:
            return {}

        first = self.steps[0]
        final = self.steps[-1]

        stats = {
            "total_steps": len(self.steps),
            "initial_population": {},
            "final_population": {},
            "population_change": {},
            "extinct_species": [],
            "surviving_species": [],
            "dominant_species": None,
        }

        # 计算各物种变化
        species_list = ["tigers", "wolves", "deer", "foxes", "rabbits"]
        final_counts = {}

        for species in species_list:
            initial = first.get(species, 0)
            final_count = final.get(species, 0)

            stats["initial_population"][species] = initial
            stats["final_population"][species] = final_count
            final_counts[species] = final_count

            if initial > 0:
                change_pct = ((final_count - initial) / initial) * 100
                stats["population_change"][species] = round(change_pct, 1)
            else:
                stats["population_change"][species] = 0

            if final_count == 0 and initial > 0:
                stats["extinct_species"].append(species)
            if final_count > 0:
                stats["surviving_species"].append(species)

        # 优势物种
        if final_counts:
            stats["dominant_species"] = max(final_counts.items(), key=lambda x: x[1])[0]

        return stats


class ChineseNarrativeGenerator:
    """中文叙事生成器"""

    # 物种中文名称映射
    SPECIES_NAMES = {
        "tigers": "老虎",
        "wolves": "灰狼",
        "deer": "梅花鹿",
        "foxes": "狐狸",
        "rabbits": "野兔",
        "ecosystem": "生态系统",
    }

    # 开头模板
    OPENINGS = [
        "在这片广袤的丛林中，",
        "岁月流转，生态的画卷缓缓展开，",
        "经过数百轮的生存博弈，",
        "大自然的选择力量在此显现：",
    ]

    # 结尾模板
    ENDINGS = [
        "这就是自然选择的力量，是主动推理驱动下的生命史诗。",
        "生态的故事仍在继续，每一个决策都塑造着种群的命运。",
        "从诞生到消亡，每一个物种都在这片土地上留下了自己的痕迹。",
    ]

    # 事件模板映射
    EVENT_TEMPLATES = {
        EventType.BIRTH: [
            "第{step}步：新的生命降临！{description}",
            "在第{step}轮，{description}",
        ],
        EventType.DEATH: [
            "第{step}步：死亡的阴影笼罩，{description}",
        ],
        EventType.PREDATION: [
            "第{step}步：捕食关系塑造生态，{description}",
            "观察到典型的捕食者-猎物动态：{description}",
        ],
        EventType.POPULATION_RISE: [
            "第{step}步：种群繁荣！{description}",
        ],
        EventType.POPULATION_FALL: [
            "第{step}步：衰退显现，{description}",
        ],
        EventType.EXTINCTION: [
            "⚫ 第{step}步：悲剧发生！{description}",
            "⚠️ 重要事件：{description}",
        ],
        EventType.BALANCE: [
            "🌟 第{step}步：{description}",
            "✨ 生态奇迹：{description}",
        ],
        EventType.RESOURCE_CRISIS: [
            "第{step}步：资源危机！{description}",
        ],
    }

    def __init__(self, events: List[EcosystemEvent], statistics: Dict[str, Any]):
        self.events = events
        self.stats = statistics

    def _translate_species(self, species: str) -> str:
        """翻译物种名称"""
        return self.SPECIES_NAMES.get(species, species)

    def generate_executive_summary(self) -> str:
        """生成执行摘要"""
        stats = self.stats
        if not stats:
            return "模拟数据不足，无法生成摘要。"

        summary = []
        summary.append("# 📊 生态系统模拟报告\n")

        summary.append("## 📈 模拟概况\n")
        summary.append(f"- 总模拟步数：**{stats['total_steps']}** 步\n")
        summary.append(f"- 存活物种：**{len(stats['surviving_species'])}** 个\n")
        summary.append(f"- 灭绝物种：**{len(stats['extinct_species'])}** 个\n")
        if stats['dominant_species']:
            summary.append(f"- 优势物种：**{self._translate_species(stats['dominant_species'])}**\n")

        summary.append("\n## 🐾 种群变化\n")
        for species, change in stats["population_change"].items():
            initial = stats["initial_population"][species]
            final = stats["final_population"][species]
            species_cn = self._translate_species(species)

            if initial > 0:
                indicator = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                summary.append(f"- {indicator} **{species_cn}**: {initial} → {final}只 ")
                if change > 0:
                    summary.append(f"(+{change}%)")
                elif change < 0:
                    summary.append(f"({change}%)")
                summary.append("\n")
            elif final > 0:
                summary.append(f"- 🆕 **{species_cn}**: {final}只 (新物种)\n")

        if stats["extinct_species"]:
            summary.append("\n## ⚠️ 灭绝物种\n")
            for species in stats["extinct_species"]:
                summary.append(f"- ❌ {self._translate_species(species)}\n")

        return "".join(summary)

    def generate_timeline(self, max_events: int = 15) -> str:
        """生成事件时间线"""
        if not self.events:
            return "未检测到显著生态事件。"

        lines = ["## 📅 关键生态事件时间线\n"]

        # 选择重要事件
        selected_events = sorted(self.events, key=lambda e: -e.magnitude)[:max_events]
        selected_events.sort(key=lambda e: e.step)  # 按时间排序

        for event in selected_events:
            templates = self.EVENT_TEMPLATES.get(event.event_type, ["第{step}步：{description}"])
            template = templates[hash(str(event.step)) % len(templates)]
            desc = self._translate_species(event.description)
            lines.append(template.format(step=event.step, description=desc))

        return "\n".join(lines)

    def generate_insights(self) -> str:
        """生成生态系统洞察"""
        insights = ["\n## 💡 生态洞察\n"]
        stats = self.stats

        # 洞察1：捕食者猎物比
        predator_total = stats["final_population"].get("tigers", 0) + stats["final_population"].get("wolves", 0)
        prey_total = stats["final_population"].get("deer", 0) + stats["final_population"].get("rabbits", 0)

        if prey_total > 0:
            ratio = predator_total / prey_total
            if ratio < 0.1:
                insights.append("- ✅ **捕食者-猎物比良好**：捕食者数量适中，生态压力较小\n")
            elif ratio < 0.3:
                insights.append("- ⚖️ **捕食者-猎物比平衡**：比例约为{:.1f}%\n".format(ratio * 100))
            else:
                insights.append("- ⚠️ **捕食者过多**：捕食者比例高达{:.1f}%，生态可能不稳定\n".format(ratio * 100))

        # 洞察2：物种多样性
        surviving = len(stats["surviving_species"])
        if surviving >= 5:
            insights.append("- 🌳 **生物多样性极高**：所有5个物种都成功存活，生态系统非常健康\n")
        elif surviving >= 3:
            insights.append("- 🌿 **生物多样性良好**：{}个物种共存，生态系统相对稳定\n".format(surviving))
        elif surviving >= 1:
            insights.append("- 🥀 **生物多样性偏低**：仅{}个物种存活，系统较为脆弱\n".format(surviving))

        # 洞察3：繁殖策略
        if stats["final_population"].get("rabbits", 0) > stats["initial_population"].get("rabbits", 1) * 2:
            insights.append("- 🐇 **r策略成功**：野兔采用高繁殖率策略，种群大幅增长\n")

        # 洞察4：顶级捕食者状况
        if stats["final_population"].get("tigers", 0) > 0:
            insights.append("- 🐅 **顶级捕食者在位**：老虎种群存活，食物链结构完整\n")
        else:
            insights.append("- 🔄 **营养级缺失**：顶级捕食者老虎已灭绝，食物链可能重构\n")

        return "".join(insights)

    def generate_full_report(self) -> str:
        """生成完整报告"""
        report = []
        report.append(self.generate_executive_summary())
        report.append("\n" + self.generate_timeline(max_events=12))
        report.append("\n" + self.generate_insights())

        # 结尾
        import random
        report.append(f"\n---\n\n*" + random.choice(self.ENDINGS) + "*")

        return "".join(report)


def generate_ecosystem_report_from_steps(steps_data: List[Dict[str, Any]]) -> str:
    """从步骤数据生成完整的中文报告"""
    analyzer = EcosystemAnalyzer(steps_data)
    analyzer.detect_all_events()
    stats = analyzer.get_summary_statistics()

    generator = ChineseNarrativeGenerator(analyzer.events, stats)
    return generator.generate_full_report()


def generate_simple_narrative(steps_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成简单的时间线叙事，用于API返回"""
    analyzer = EcosystemAnalyzer(steps_data)
    events = analyzer.detect_all_events()
    stats = analyzer.get_summary_statistics()

    narrative_events = []
    for event in sorted(events, key=lambda e: -e.magnitude)[:10]:
        narrative_events.append({
            "step": event.step,
            "type": event.event_type.value,
            "species": event.species,
            "description": event.description,
            "importance": round(event.magnitude, 2)
        })

    return {
        "events": narrative_events,
        "statistics": stats,
    }

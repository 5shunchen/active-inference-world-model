# 主动推理世界模拟器 (Active Inference World Simulator)

一个端到端的生态系统模拟平台，基于主动推理原理，智能体通过最小化变分自由能来生存、捕食和繁殖。

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 🌟 核心特性

### 🧠 主动推理引擎
- **变分自由能最小化**：智能体通过感知和行动维持生存边界
- **贝叶斯滤波**：准确的状态估计
- **策略选择**：基于预期自由能最小化的行动选择
- **风险与模糊分解**：EFE = 风险项 + 模糊项

### 🌍 完整生态系统
| 物种 | 类型 | 角色 |
|------|------|------|
| 🐅 **老虎** | 顶级捕食者 | 高能量需求，强大捕食能力 |
| 🐺 **狼** | 群居捕食者 | 中等能量需求，群体协作 |
| 🦌 **鹿** | 草食动物 | 警惕性高，快速繁殖 |
| 🦊 **狐狸** | 机会主义 | 小型猎物专家，适应性强 |
| 🐇 **兔子** | 小型草食动物 | 高繁殖率，种群波动大 |

### 🏞️ 动态环境
- **4个生态区域**：猎物区、水区、安全区、草地
- **昼夜循环**：影响能见度和捕食成功率
- **天气系统**：晴朗、多云、雨天、雾天
- **资源再生**：水、食物随时间再生

### 📊 可视化与API
- **实时仪表盘**：种群动态折线图，5物种同步追踪
- **模拟历史**：所有模拟持久化存储，支持回放
- **REST API**：完整的FastAPI接口，自动生成文档
- **命令行工具**：一键启动模拟、服务、测试

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 推荐使用虚拟环境

### 安装依赖
```bash
# 克隆项目
git clone https://github.com/5shunchen/active-inference-world-model.git
cd active-inference-world-model

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行方式

#### 1. 启动Web仪表盘（推荐）
```bash
python -m src.cli serve
```
访问 http://localhost:8000 即可使用可视化控制面板

#### 2. 命令行运行模拟
```bash
# 完整生态系统模拟
python -m src.cli run-ecosystem \
    --tigers 2 \
    --wolves 3 \
    --deer 15 \
    --foxes 5 \
    --rabbits 20 \
    --steps 500

# 老虎生命故事模式
python -m src.cli run-tiger \
    --prompt "一只老虎的丛林生存故事" \
    --steps 300
```

#### 3. 运行全部测试
```bash
python -m pytest tests/ -v
```

#### 4. Docker部署
```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f api
```

---

## 📡 API 接口

### 模拟管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/ecosystem` | 启动生态系统模拟 |
| `POST` | `/api/lifestory` | 生成生命故事模拟 |
| `GET` | `/api/results/{id}` | 获取模拟结果（内存） |
| `GET` | `/api/simulations` | 列出全部历史模拟 |
| `GET` | `/api/simulations/{id}/detail` | 获取模拟详情 |
| `GET` | `/api/simulations/{id}/timeseries` | 获取种群时间序列数据 |
| `DELETE` | `/api/simulations/{id}` | 删除模拟记录 |

### 系统接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 仪表盘首页 |
| `GET` | `/docs` | Swagger API文档 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/config` | 可用配置选项 |

### 请求示例
```bash
# 启动生态系统模拟
curl -X POST http://localhost:8000/api/ecosystem \
    -H "Content-Type: application/json" \
    -d '{
        "n_tigers": 2,
        "n_wolves": 3,
        "n_deer": 15,
        "n_foxes": 5,
        "n_rabbits": 25,
        "max_steps": 500,
        "enhanced_mode": true
    }'

# 查询种群时间序列
curl http://localhost:8000/api/simulations/{id}/timeseries
```

---

## 🎯 系统架构

### 模块依赖关系
```
                    ┌─────────────────────┐
                    │   distributions.py  │
                    │   (概率分布工具)    │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  generative_model.py│
                    │   (生成模型抽象)    │
                    └─────────┬───────────┘
                              │
    ┌─────────────────────────┼──────────────────────────┐
    │                         │                          │
┌───▼─────┐            ┌─────▼──────┐             ┌───▼─────┐
│  world  │            │ inference  │             │ language │
│ simulator│            │  engine    │             │  parser  │
│  (环境)  │            │ (推理引擎) │             │ (语言解析)│
└────┬─────┘            └─────┬──────┘             └─────┬───┘
     │                        │                          │
     │                  ┌─────▼──────┐             ┌────▼─────┐
     │                  │ thermostat │             │  scene    │
     │                  │  (恒温器)  │             │  builder  │
     │                  └────────────┘             └─────┬─────┘
     │                                                   │
┌────▼──────────┐                              ┌────────▼───────┐
│   ecosystem    │                              │  tiger_ecology │
│  (5物种子系统)  │                              │    (老虎模型)   │
└────┬──────────┘                              └────────┬───────┘
     │                                                   │
     └──────────────────────┬────────────────────────────┘
                           │
                  ┌────────▼──────────┐
                  │  life_story_api.py │
                  │  (入口API封装)    │
                  └────────┬──────────┘
                           │
                    ┌──────▼───────┐
                    │  api.py       │
                    │  (FastAPI)    │
                    └──────┬───────┘
                    ┌──────▼───────┐
                    │  cli.py       │
                    │  (命令行)      │
                    └───────────────┘
```

### 可视化与存储层
```
┌─────────────────┐    ┌─────────────────┐
│ visual_renderer │    │   database.py   │
│  (Matplotlib)   │    │  (SQLAlchemy)   │
└────────┬────────┘    └────────┬────────┘
         │                       │
┌────────▼────────┐    ┌────────▼────────┐
│ narrative_gen.py │    │  dashboard.html │
│  (叙事生成器)    │    │  (Chart.js图表)  │
└─────────────────┘    └─────────────────┘
```

---

## 📊 项目状态

| 指标 | 状态 |
|------|------|
| ✅ 测试覆盖 | 81 个测试全部通过 |
| ✅ 核心模块 | 13个独立模块 |
| ✅ 动物物种 | 5种（含完整性状模型） |
| ✅ API端点 | 9个REST接口 |
| ✅ Docker支持 | 完全就绪 |
| ✅ 数据库持久化 | SQLite |
| ✅ 实时可视化 | Chart.js动态图表 |
| 📦 GitHub仓库 | [5shunchen/active-inference-world-model](https://github.com/5shunchen/active-inference-world-model) |
| 🚀 版本 | v2.0.0 |

---

## 📁 项目结构

```
active-inference-world-model/
├── src/
│   ├── distributions.py       # 概率分布（分类、多变量高斯）
│   ├── generative_model.py    # 生成模型抽象基类
│   ├── world_simulator.py     # 世界模拟器（隐藏真实状态）
│   ├── inference_engine.py     # 主动推理引擎（EFE、贝叶斯滤波）
│   ├── thermostat.py           # 恒温器演示模型
│   ├── language_parser.py      # 自然语言提示解析器
│   ├── scene_builder.py        # 场景配置→模型构建器
│   ├── tiger_ecology_model.py  # 老虎生态模型
│   ├── ecosystem.py            # 5物种完整生态系统
│   ├── enhanced_ecosystem.py   # 增强型生态系统模拟器
│   ├── visual_renderer.py      # 视频渲染器（Matplotlib + OpenCV）
│   ├── narrative_generator.py  # 事件检测与叙事生成
│   ├── database.py             # SQLAlchemy数据库层
│   ├── api.py                  # FastAPI服务
│   ├── cli.py                  # 命令行入口
│   └── templates/
│       └── dashboard.html      # 可视化仪表盘
├── tests/                      # 81个单元测试
├── requirements.txt            # 依赖清单
├── Dockerfile                  # Docker镜像构建
├── docker-compose.yml          # Compose编排
├── .gitignore                  # Git忽略配置
└── CLAUDE.md                   # 项目说明文档
```

---

## 🧪 开发路线图

### ✅ Phase 1 - 核心引擎（已完成）
- 主动推理核心算法实现
- 概率分布工具库
- 世界模拟器（状态隐藏）
- 恒温器概念验证

### ✅ Phase 2 - 生命故事（已完成）
- 自然语言提示解析
- 老虎生态系统模型
- 叙事生成与字幕输出
- 视频帧渲染与MP4生成

### ✅ Phase 3 - 生态扩展（已完成）
- 5物种种群动态
- 捕食者-猎物交互
- 天气与昼夜循环
- 资源竞争与再生

### ✅ Phase 4 - 平台化（已完成）
- SQLite数据库持久化
- Chart.js实时可视化
- 历史模拟回放
- Docker容器化

### 🚧 Phase 5 - AI增强（计划中）
- LLM驱动的叙事生成
- GPT生态系统描述
- 智能体策略学习
- 迁移学习支持

### 🚧 Phase 6 - 扩展生态（计划中）
- 更多物种：鸟类、鱼类、昆虫
- 植物种群动态
- 季节更替系统
- 地形与栖息地建模

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支
3. 确保所有测试通过
4. 提交Pull Request

---

## 📄 许可证

MIT License - 详见 LICENSE文件

---

## 🙏 致谢

本项目基于Karl Friston的自由能原理（Free Energy Principle）构建，
致力于展示主动推理在复杂生态系统建模中的强大潜力。

---

**Enjoy simulating emergent ecosystems with active inference!** 🐅🌍🦌

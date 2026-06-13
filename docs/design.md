# Dota 2 Copilot — 设计文档 v0.1

> 目标：在打 Dota 2 时，通过**采集小地图画面**实时感知英雄动向，按可配置规则发出**语音 / 文字提醒**，帮助玩家做出更好的决策（gank 预警、被 gank 风险、并线安全等）。
>
> 范围限定：**只使用屏幕上肉眼可见的信息**，不读内存、不注入、不模拟操作 —— 完全合规、不触发 VAC。

---

## 1. 设计目标与非目标

### 1.1 目标
- **G1**：实时识别小地图上敌方/友方英雄位置（精度：< 1 个英雄图标半径）
- **G2**：维护过去 `N` 秒的位置历史窗口（`N` 可配置）
- **G3**：基于历史窗口与基础地图信息，给出以下提醒：
  - **A1 Gank 预警**：敌方多人向某条路集结 / 多人消失视野
  - **A2 被 gank 风险**：友方英雄位置过于深入 + 附近有敌方威胁
  - **A3 并线安全评级**：每条线当前是否安全（低/中/高/极高风险）
  - **A4（Optional）野区刷野风险评级**：进野前提示
- **G4**：提醒通道**可插拔可配置**：Windows 原生 Toast + 中文 TTS 语音
- **G5**：决策引擎**可插拔**：规则 / 本地模型 / 远端 LLM 三种后端可切换

### 1.2 非目标（明确不做）
- ❌ 不做雾里位置预测的"全图透视"（违反精神，且无法做到）
- ❌ 不做任何自动按键 / 自动操作
- ❌ 不解析游戏内存或网络包
- ❌ 不在 P1 阶段做敌方装备识别（属于 P2/P3）

### 1.3 运行环境
- **目标运行平台**：Windows 10/11（玩 Dota 的机器）
- **开发环境**：当前 workspace 在 Linux/WSL，跨平台代码 + Windows 专属 notifier 分离
- **Python 版本**：3.11+
- **不强依赖 GPU**（小地图识别用经典 CV 即可；如果后期上深度模型再说）

---

## 2. 总体架构

```
┌────────────────────────────────────────────────────────────────┐
│                       Dota 2 Copilot                            │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  采集层      │───▶│  状态管理层   │───▶│   决策层          │  │
│  │  Capture    │    │  StateStore  │    │   Decision        │  │
│  │             │    │  (滑动窗口)   │    │   ┌──────────┐    │  │
│  │ - 截屏       │    │              │    │   │ Rule     │    │  │
│  │ - 小地图识别  │    │ - 帧序列     │    │   │ ML       │    │  │
│  │ - 英雄定位    │    │ - 轨迹       │    │   │ LLM      │    │  │
│  │ - 兵线识别    │    │ - 事件       │    │   └──────────┘    │  │
│  └─────────────┘    └──────────────┘    └────────┬─────────┘  │
│         ▲                   ▲                    │             │
│         │                   │                    ▼             │
│  ┌──────┴──────┐    ┌──────┴──────┐    ┌─────────────────┐    │
│  │  校准 &      │    │  地图知识库   │    │   通知层         │    │
│  │  配置        │    │  MapKB       │    │   Notifier      │    │
│  │             │    │              │    │                 │    │
│  │ - 小地图区域  │    │ - 兵线路径   │    │ - Win Toast     │    │
│  │ - 英雄颜色    │    │ - 野点       │    │ - 中文 TTS      │    │
│  │ - 帧率/N 秒   │    │ - 关键点     │    │ - 控制台        │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**数据流**：屏幕 → 采集层（识别为帧）→ 状态管理层（滑窗 + 轨迹）→ 决策层（产出 Alert）→ 通知层（用户感知）

**关键设计原则**：
- **采集 / 决策 / 通知三层解耦**：每层只通过定义好的数据结构通信
- **决策策略可插拔**：通过抽象接口 `DecisionEngine`，规则 / ML / LLM 都是它的实现
- **配置驱动**：N 秒、采样频率、启用哪些规则、用哪个决策后端、开哪些通知通道，全部走 yaml

---

## 3. 模块设计

### 3.0 地图知识库（MapKB）

**职责**：内置 Dota 2 地图的静态信息，给决策层用。

**核心数据**（所有坐标使用**小地图归一化坐标 0.0–1.0**，便于跨分辨率）：

```python
class MapKB:
    # 三条兵线的关键路径点（折线），分别近天辉/夜魇
    lanes: dict[Lane, list[Point]]   # Lane.TOP / MID / BOT
        # 每条线由 N 个关键点组成（折线），形如：
        # top: [(0.05, 0.95), (0.05, 0.5), (0.1, 0.1), ...]

    # 野点位置（按队伍分大野/小野/远古）
    camps: dict[Side, list[Camp]]
        # Camp: id, type(small/medium/large/ancient), pos, owner_side

    # 关键地标
    landmarks: dict[str, Point]
        # roshan_pit, rune_top, rune_bot, secret_shop, ...

    # 河道与塔位置
    rivers: list[Polygon]
    towers: dict[Side, dict[Lane, list[Point]]]   # T1/T2/T3
```

**实现思路**：
- 第一版：手工标定一次（在一张高清小地图截图上点击采集坐标）
- 提供一个标定脚本 `scripts/calibrate_mapkb.py`
- 存为 `assets/mapkb.json`，运行时加载

**为什么用归一化坐标**：玩家屏幕分辨率不同，小地图尺寸不同；归一化后 MapKB 一份通用，运行时再乘上当前小地图像素尺寸。

---

### 3.1 采集层（Capture）

#### 3.1.1 截屏
- 库：[`mss`](https://github.com/BoboTiG/python-mss)（跨平台、零拷贝、~1ms/帧）
- 频率：默认 1 Hz，可调（最高建议 ≤ 10 Hz，CV 处理耗时是瓶颈）

#### 3.1.2 小地图区域定位
- **一次性校准**：用户运行 `dota2-copilot calibrate`，在截图上框选小地图区域
- 存到 `config/minimap.json`：`{x, y, width, height}`
- 后续每帧只截这一块（更快）

#### 3.1.3 英雄检测
**两阶段方案**：

**Stage 1 — 颜色分割（必做，P1 完成）**
- 小地图上每个英雄是一个**带彩色圆点 + 队伍色边框**的图标
- 敌方边框红色（HSV 大致 `H∈[0,10]∪[170,180]`），友方绿色（`H∈[40,80]`）
- 自己有特殊高亮（白边或方框）
- 步骤：
  1. BGR → HSV
  2. 按颜色范围做 mask
  3. 形态学开运算去噪
  4. `cv2.connectedComponentsWithStats` 找连通域
  5. 过滤面积（图标大致 14×14 ~ 18×18 px）
  6. 输出每个 blob 的中心点 → `HeroBlob(team, pos, area)`

**Stage 2 — 英雄身份识别（P2 增强，可选）**
- 用每个英雄的小地图头像做模板匹配（`cv2.matchTemplate`，归一化相关）
- 素材：从游戏文件 / 社区 wiki 提取每个英雄的 minimap icon
- 输出：`HeroIdentified(team, pos, hero_id, confidence)`

> P1 阶段：只识别**敌方红点数量与位置**就足够驱动 A1/A2/A3。具体是谁可以先不知道。

#### 3.1.4 兵线识别（P2，先占位）
- 兵线在小地图上是**密集的细小红/绿点群**（小兵图标）
- 思路：在颜色分割后，识别"密集小点集群"而非"单个大点"
- 输出每条线上的兵线推进位置（沿兵线折线的进度 0~1）
- **P1 阶段可先用"无兵线数据"，仅基于英雄位置评估**

#### 3.1.5 输出数据结构

```python
@dataclass
class Frame:
    timestamp: float                    # 采集时间（秒）
    game_time: float | None             # 游戏时间（暂无来源，预留）
    enemies: list[HeroBlob]             # 检测到的敌方
    allies: list[HeroBlob]              # 检测到的友方（含自己）
    self_pos: Point | None              # 自己的位置（特殊标记识别）
    creeps: dict[Lane, float] | None    # 兵线推进 0~1（P2）
    raw_minimap: np.ndarray | None      # 可选保留原图（调试用）
```

---

### 3.2 状态管理层（StateStore）

**职责**：维护滑动窗口 + 简单轨迹关联。

```python
class StateStore:
    window_seconds: float               # N 秒，可配
    frames: deque[Frame]                # 时间序列

    def push(self, frame: Frame): ...
    def recent(self, seconds: float) -> list[Frame]: ...
    def tracks(self) -> list[Track]: ...
        # 跨帧关联同一个英雄的轨迹（基于位置邻近 + 颜色 + 身份）
```

**轨迹关联（P1 简化版）**：
- 假设敌方最多 5 个红点，按"最近邻 + 上一帧位置 < 阈值"做匹配
- 对消失的红点保留 `last_seen_at` 与 `last_pos`
- 输出：`Track(team, current_pos, history: list[(t, pos)], last_seen, missing_for)`

**派生事件**（StateStore 自动产出，供决策层消费）：
- `HeroDisappeared(track_id, last_pos, t)`
- `HeroAppeared(track_id, pos, t)`
- `HeroMoved(track_id, from, to, speed)`
- `MultiEnemyConverging(positions, target_area)`（多个敌方向同一区域移动）

---

### 3.3 决策层（Decision）

#### 3.3.1 抽象接口

```python
class DecisionEngine(Protocol):
    def evaluate(self, state: StateStore, kb: MapKB, cfg: Config) -> list[Alert]:
        ...

@dataclass
class Alert:
    level: Literal["info", "warn", "danger", "critical"]
    category: Literal["gank", "ally_risk", "lane_safety", "jungle_risk"]
    title: str                  # 简短标题，给 toast 用
    message: str                # 详细描述
    speech: str                 # 语音播报（可与 message 不同，更口语化）
    target_lane: Lane | None
    target_hero: HeroId | None
    confidence: float           # 0~1
    cooldown_key: str           # 用于去重 / 冷却（避免 5 秒内重复同一警报）
```

#### 3.3.2 三种后端

| 后端 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| **RuleEngine** | 快、可控、零成本、可解释 | 表达力有限，调参靠人 | **P1 必做，主力** |
| **MLEngine** | 能学复杂模式（轨迹 → gank 概率） | 需要数据集、训练流程 | P3 增强 |
| **LLMEngine** | 表达力强、能给推理过程 | 延迟（>1s）、有成本、可能瞎说 | P2 实验，做"慢思考"层 |

#### 3.3.3 推荐：**混合策略**

```
高频（1 Hz）：RuleEngine 快速产出 info/warn
              ↓
触发条件满足时（如 warn 持续 3 秒）→ 异步调用 LLMEngine 做"二次确认"，
                                      可能升级为 danger 或撤销
```

这样：响应快（规则）+ 智能（LLM 兜底），且 LLM 调用频率可控（成本可控）。

#### 3.3.4 P1 内置规则（举例）

| 规则 ID | 触发条件 | 输出 |
|---|---|---|
| `R-GANK-01` | 敌方 ≥ 3 人 5 秒内消失视野 | warn: 可能集结 gank |
| `R-GANK-02` | 敌方 ≥ 2 人正在向某友方靠近且距离 < 阈值 | danger: gank 即将发生 |
| `R-ALLY-01` | 友方位于敌方半场 + 附近 800 内有 ≥ 2 敌方 | danger: 友方有被 gank 风险 |
| `R-LANE-01` | 某条线上敌方人数 - 友方人数 ≥ 2 | warn: 该线不要去 |
| `R-LANE-02` | 某条线 30 秒内无敌方出现 | info: 安全线 |
| `R-JUNGLE-01` | 进入野区前，最近 N 秒该野区附近见过敌方 | warn: 野区有风险 |

每条规则的阈值（距离、人数、秒数）都放在 `config/rules.yaml`，用户可调。

#### 3.3.5 决策上下文（喂给 LLM 时的结构化输入示例）

```json
{
  "now": 632.5,
  "window_seconds": 15,
  "self": {"pos": [0.45, 0.52], "lane": "mid"},
  "allies_visible": [
    {"id": "ally_1", "pos": [0.2, 0.85], "lane": "bot"}
  ],
  "enemies_tracks": [
    {"id": "e1", "last_seen": 628.0, "last_pos": [0.55, 0.48], "missing_for": 4.5},
    {"id": "e2", "last_seen": 627.0, "last_pos": [0.6, 0.5], "missing_for": 5.5},
    {"id": "e3", "current_pos": [0.65, 0.55], "trajectory": [[630, [0.7,0.6]], [631, [0.68,0.58]]]}
  ],
  "map_context": {
    "mid_rune_in": 12,
    "ally_bot_distance_to_t1": 0.05
  }
}
```

LLM 输出：JSON 格式的 `Alert` 列表（用 system prompt 强约束）。

---

### 3.4 通知层（Notifier）

#### 3.4.1 通道

| 通道 | 库 / 接口 | 备注 |
|---|---|---|
| **Windows Toast** | [`windows-toasts`](https://github.com/DatGuy1/Windows-Toasts) | 调 WinRT `ToastNotificationManager`，原生 Banner |
| **中文 TTS** | 见下方对比 | 默认开 |
| **控制台** | 标准输出 | 调试用 |
| **Overlay**（未来） | PyQt 透明窗 / 单独进程 | P3 可选 |

#### 3.4.2 Windows Toast 实现要点
- `windows-toasts` 库封装了 WinRT，比 `win10toast`（已废弃）更稳定
- 支持：标题 + 正文 + 图标 + 按钮 + 自动消失时间
- 需要给应用一个 AUMID（应用模型 ID）——库会自动处理
- 紧急程度映射：
  - `info`: 静默 toast（不出声）
  - `warn`: 普通 toast
  - `danger / critical`: 重要 toast + 震动音效

#### 3.4.3 中文 TTS 方案对比

| 方案 | 离线 | 中文质量 | 依赖 | 推荐度 |
|---|---|---|---|---|
| **Windows SAPI**（系统自带） | ✅ | 中（机械感） | 无 | ★★★ 默认 |
| **edge-tts**（微软在线） | ❌ | **优秀** | 联网 | ★★★★ 推荐 |
| **pyttsx3**（封装 SAPI） | ✅ | 中 | 无 | ★★ |
| **本地 VITS / GPT-SoVITS** | ✅ | 优秀，可换音色 | 需 GPU、模型大 | ★★ P3 |
| 第三方 TTS（科大讯飞、阿里） | ❌ | 优秀 | 需 API Key | ★★ |

**P1 默认**：edge-tts（中文自然，免费，约 200ms 延迟）+ SAPI 兜底（离线时）
**音色**：默认 `zh-CN-XiaoxiaoNeural`（女声）或 `zh-CN-YunxiNeural`（男声），可配

#### 3.4.4 防打扰策略
- **冷却**：每个 `cooldown_key` 在 N 秒内只通知一次
- **抑制**：critical 级别会抑制同时刻的 info/warn
- **静音模式**：用户可一键关闭语音 / Toast（保留控制台）

#### 3.4.5 配置示例

```yaml
notifier:
  channels:
    toast:
      enabled: true
      min_level: info
    tts:
      enabled: true
      min_level: warn       # 只对 warn 及以上播报
      engine: edge-tts
      voice: zh-CN-YunxiNeural
      rate: "+0%"
      fallback: sapi
    console:
      enabled: true
  cooldown:
    default_seconds: 5
    by_category:
      gank: 8
      lane_safety: 15
```

---

## 4. 配置文件总览

```
config/
├── app.yaml          # 全局：采样频率、N 秒窗口、决策后端选择
├── minimap.json      # 校准结果：小地图屏幕坐标
├── rules.yaml        # 规则参数（阈值、开关）
├── notifier.yaml     # 通知配置
└── llm.yaml          # （可选）LLM 后端：模型、API Key、prompt
```

**`app.yaml` 示例**：
```yaml
capture:
  fps: 1                       # 采样频率
  history_seconds: 30          # N 秒窗口
decision:
  backend: rule                # rule | llm | hybrid
  hybrid:
    promote_to_llm_after: 3.0  # warn 持续 3 秒后让 LLM 复核
```

---

## 5. 目录结构（提议）

```
dota2_copilot/
├── pyproject.toml
├── README.md
├── docs/
│   ├── design.md                  # 本文件
│   └── calibration.md             # 校准操作手册
├── config/                        # 用户可改
│   ├── app.yaml
│   ├── minimap.json               # 校准产物，gitignore
│   ├── rules.yaml
│   └── notifier.yaml
├── assets/                        # 内置素材
│   ├── mapkb.json                 # 地图知识库
│   └── hero_icons/                # 英雄小地图头像（P2）
├── src/dota2_copilot/
│   ├── __init__.py
│   ├── cli.py                     # 入口：run / calibrate / debug
│   ├── config.py                  # pydantic-settings 加载
│   ├── types.py                   # 共享数据类型
│   │
│   ├── capture/
│   │   ├── screen.py              # mss 封装
│   │   ├── minimap.py             # 小地图区域 + 颜色分割
│   │   ├── hero_detect.py         # 英雄 blob 检测
│   │   └── identify.py            # 模板匹配识别英雄（P2）
│   │
│   ├── state/
│   │   ├── store.py               # StateStore 滑窗
│   │   ├── tracker.py             # 简单 nearest-neighbor tracker
│   │   └── events.py              # 派生事件
│   │
│   ├── mapkb/
│   │   ├── loader.py
│   │   └── geometry.py            # 距离 / 是否在线上 / 是否在野区
│   │
│   ├── decision/
│   │   ├── base.py                # DecisionEngine 抽象
│   │   ├── rule_engine.py         # P1 主力
│   │   ├── llm_engine.py          # P2
│   │   ├── hybrid.py              # P2
│   │   └── rules/
│   │       ├── gank.py
│   │       ├── ally_risk.py
│   │       ├── lane_safety.py
│   │       └── jungle_risk.py
│   │
│   ├── notifier/
│   │   ├── base.py
│   │   ├── toast_win.py           # Windows Toast
│   │   ├── tts.py                 # edge-tts + SAPI
│   │   ├── console.py
│   │   └── manager.py             # 多通道分发 + 冷却
│   │
│   └── tools/
│       ├── calibrate_minimap.py   # 框选小地图
│       ├── calibrate_mapkb.py     # 点击采集地图关键点
│       ├── debug_preview.py       # 实时显示检测结果（OpenCV 窗口）
│       └── snapshot.py
│
├── scripts/                       # 一次性脚本
│   └── fetch_hero_icons.py        # 从社区下载英雄小地图图标
│
└── tests/
    ├── fixtures/                  # 测试用截图
    ├── test_minimap.py
    ├── test_rule_engine.py
    └── test_tracker.py
```

---

## 6. 实施路线图

### Milestone 1 — 看见英雄（1–2 天工作量）
- [ ] 项目骨架、配置加载、CLI
- [ ] mss 截屏 + 小地图区域校准工具
- [ ] HSV 颜色分割识别敌/友方红绿点
- [ ] 调试预览窗口（用 OpenCV 在原图上画检测框）
- **验收**：实战中能在控制台稳定输出每秒一帧的敌方红点数与坐标

### Milestone 2 — 状态与基础规则（1–2 天）
- [ ] MapKB 数据结构 + 手工标定一份 `mapkb.json`
- [ ] StateStore + 最近邻 tracker
- [ ] R-GANK-01 / R-ALLY-01 / R-LANE-01 三条规则
- [ ] Console notifier
- **验收**：在自己实战录像 / 直播视频上重放，能合理触发 gank 警报

### Milestone 3 — Windows 提醒落地（1 天）
- [ ] Windows Toast notifier
- [ ] edge-tts TTS notifier（中文）
- [ ] 冷却 / 抑制 / 配置开关
- **验收**：实战中收到的提醒不嘈杂、不漏关键事件

### Milestone 4 — 完善规则与并线评级（2 天）
- [ ] R-LANE-02、R-JUNGLE-01
- [ ] 并线安全综合评级（多规则加权 → 4 档）
- [ ] 配置热加载（改 yaml 不用重启）

### Milestone 5（可选）— 智能化
- [ ] LLM Engine（OpenAI / 本地 ollama 任选）
- [ ] Hybrid 策略
- [ ] 英雄身份识别（模板匹配）→ 提醒更具体（"敌方斧王在中路消失"）

### Milestone 6（可选）— 体验提升
- [ ] 兵线识别
- [ ] PyQt overlay（透明窗显示提醒，避免 Toast 错过）
- [ ] 录制 + 离线回放模式（用于调参 / 复盘）

---

## 7. 关键技术决策与权衡

| 决策点 | 选择 | 理由 |
|---|---|---|
| 截屏库 | `mss` | 跨平台、快、零依赖 |
| 图像处理 | OpenCV（经典 CV） | 小地图识别用色彩分割足够，无需深度学习 |
| 坐标系 | 小地图归一化 (0~1) | 跨分辨率通用 |
| 配置语言 | YAML + Pydantic | 易读、有 schema 校验 |
| 异步模型 | 主循环单线程 + TTS 异步 | 简单可控；CV 单帧 < 50ms，1 Hz 够用 |
| Toast 库 | `windows-toasts` | 现代 WinRT，比 `win10toast` 维护更好 |
| TTS 默认 | `edge-tts` | 中文音质明显优于 SAPI |
| 决策架构 | 可插拔 + 默认规则 | P1 快速可用，P2/P3 平滑升级 |

---

## 8. 风险与限制

| 风险 | 影响 | 缓解 |
|---|---|---|
| 小地图缩放/位置变化 | 识别失效 | 校准工具支持随时重新校准；启动时做一致性自检 |
| 玩家自定义小地图颜色（色盲模式） | 颜色阈值失效 | 在 `rules.yaml` 暴露 HSV 阈值；预置色盲模式预设 |
| 1 Hz 采样错过快速移动 | 漏检 | 可调高到 5 Hz；关键事件用"消失"事件而非"持续追踪" |
| 烟雾/影魔黑洞等遮挡小地图局部 | 短暂误检 | tracker 的 `missing_for` 容忍度可调 |
| 误报扰民 | 体验差 | 冷却 + 抑制 + 按级别分通道 + 可一键静音 |
| LLM 幻觉 | 错误建议 | 仅做"二次确认"，不能凭空创造新警报；严格 JSON schema |
| 跨平台开发（Linux workspace） | Windows-only 模块无法在 Linux 测 | notifier 抽象层 + Linux 提供 `notify-send` 兜底，便于本地开发 |

---

## 9. 已确认的运行环境与策略

| 项 | 决定 |
|---|---|
| 屏幕分辨率 | **2560 × 1440（2K）** |
| 小地图位置 | **左下角**（默认 HUD） |
| Python 版本 | **3.11+** |
| P1 决策后端 | **纯规则**先把链路打通；`DecisionEngine` 抽象就位，LLM 后续无缝接入 |
| 通知通道（P1） | **仅 Windows Toast Banner**；TTS 模块预留接口但不集成 |
| TTS 后续偏好 | 优先**本地**方案（SAPI / pyttsx3 / 本地 VITS），edge-tts 作为备选 |
| 录像回放模式 | **要**——采集阶段可选保存每帧截图 + 检测结果，便于离线调参 |

### 2K 分辨率下的小地图参数（初值，留待校准时微调）
- 小地图默认在屏幕左下，参考像素区域：约 `x∈[0, 340], y∈[1100, 1440]`（具体由 calibrate 工具确定）
- 英雄图标在 2K 下大约 `18–22 px` 直径 → 颜色分割面积过滤阈值 `area ∈ [80, 500]`
- 这些值写入 `config/minimap.json`，不写死在代码里

---

## 10. 下一步（Milestone 1 任务清单）

1. Python 项目骨架（`pyproject.toml` + src layout，依赖：`mss`、`opencv-python`、`numpy`、`pydantic`、`pydantic-settings`、`pyyaml`、`typer`、`loguru`）
2. `capture/screen.py`：mss 封装（区域截屏）
3. `tools/calibrate_minimap.py`：交互式框选小地图区域，写入 `config/minimap.json`
4. `capture/minimap.py`：HSV 颜色分割 + 连通域 → `HeroBlob`
5. `tools/debug_preview.py`：实时窗口显示原图 + 检测框 + 红/绿点统计
6. `tools/record.py`：录像回放模式 —— 把每帧 minimap 截图与检测结果（JSON）落盘到 `recordings/<session>/`

> Linux 开发友好性：以上模块均可在 Linux 上运行调试（用录像 / 静态图测试）；Windows-only 的 notifier 在 M3 才引入，并通过 `notifier/base.py` 抽象隔离，本地用 console notifier 兜底。

完成 M1 后端到端能看到："**我截屏 → 找到 N 个红点 → 在窗口中高亮显示**"，接下来叠加状态管理 / 规则 / Toast 就是顺水推舟。

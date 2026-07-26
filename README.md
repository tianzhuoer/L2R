# L2R: Language-to-Reinforcement Learning for Thermocline Tracking

L2R 使用语言模型驱动的温跃层深度预测，为 Attention DQN 提供奖励塑形信号，使 AUV
能够在动态 CTD 温盐剖面中搜索、接近并持续采样温跃层。

## 方法

```text
CTD 温盐深数据
      │
      ▼
温跃层跟踪环境 ── 最近 20 步局部 T/S 观测 ──► Attention DQN ──► AUV 动作
      │                                              │
      └──► Reprogram-TSF + LLM ─► 温跃层深度预测 ──► 奖励塑形
                                                     │
预训练 Attention DQN ───────────────────────────────► KL 稳定项
```

训练目标由两部分组成：

1. Double-DQN 的 TD 损失，用于学习任务策略。
2. 当前策略与预训练参考策略之间的退火 KL 正则，用于降低微调初期的策略漂移。

LLM/TSF 预测不直接替代 RL 策略，而是在 AUV 完成一个升沉 leg 时，根据轨迹是否穿越
预测温跃层区间、leg 中心与预测深度的距离及覆盖跨度生成奖励。

## 目录结构

```text
.
├── train.py                    # L2R 主训练入口
├── config.py                   # RL、环境、路径和日志配置
├── thermocline_env.py          # 温跃层跟踪 Gymnasium 环境
├── generate_split_manifest.py  # CTD 文件级数据划分
├── requirements.txt
├── .env.example
│
├── models/
│   └── attention_dqn.py        # Attention DQN
├── utils/
│   ├── ctd_data.py             # CTD 数据读取和采样
│   └── replay_memory.py        # 序列经验回放
├── llm_tsf/
│   ├── configs/
│   │   ├── shared.py
│   │   └── reprogram.py
│   └── models/
│       └── reprogram_backbone.py
│
├── data/                       # 本地 CTD 数据，不提交
├── checkpoints/                # 预训练 DQN、LoRA 和回归头，不提交
└── modelsave/                  # 新训练权重，不提交
```

## Conda 环境

本项目已使用现有 Conda 环境 `env11` 验证，实际版本基线为：

```text
Python 3.11.13
PyTorch 2.8.0+cu128
CUDA runtime 12.8
```

在当前开发机器上直接使用：

```powershell
conda activate env11
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python train.py
```

如果需要在另一台机器创建独立环境，使用从 `env11` 提炼出的最小配置：

```bash
conda env create -f environment.yml
conda activate l2r
```

`environment.yml` 和 `requirements.txt` 中的核心包版本均来自当前可正常导入 L2R 的
`env11`，没有导出该环境内与本项目无关的数百个工具包。

若目标机器不支持 CUDA 12.8，请先根据该机器的 CUDA/驱动安装匹配的 PyTorch，再安装
其余依赖；此时不要强行安装 `torch==2.8.0+cu128`。

仅在已有兼容 Python 环境中使用 pip 时：

```bash
pip install -r requirements.txt
```

## 数据

训练环境读取包含以下字段的 CTD MATLAB 文件：

- 深度网格
- 温度剖面
- 盐度剖面
- MATLAB datenum 时间网格

默认数据目录是 `data/ctd`。也可以设置：

```powershell
$env:L2R_CTD_DIR = "D:\datasets\CTD"
$env:L2R_SPLIT_MANIFEST = "D:\datasets\CTD\split_manifest.json"
```

Linux：

```bash
export L2R_CTD_DIR=/data/CTD
export L2R_SPLIT_MANIFEST=/data/CTD/split_manifest.json
```

生成文件级数据划分：

```bash
python generate_split_manifest.py
```

manifest 会将文件分到 `rl_train`、`llm_train`、`llm_val`、`llm_test`、
`validation` 和 `test`，用于隔离 RL 与 LLM 训练数据。

## 检查点

需要准备三类权重：

```text
checkpoints/
├── pretrained_attention_dqn.pth
├── llm/
│   └── base_model/             # Hugging Face 基座模型完整目录
└── reprogram/
    ├── lora_best/
    │   ├── adapter_config.json
    │   └── model.safetensors
    └── head_best.pt
```

路径可以通过环境变量覆盖：

```powershell
$env:L2R_PRETRAIN_MODEL = "D:\models\attention_dqn.pth"
$env:L2R_LORA_DIR = "D:\models\reprogram\lora_best"
$env:L2R_HEAD_PATH = "D:\models\reprogram\head_best.pt"
$env:L2R_BASE_MODEL = "checkpoints\llm\base_model"
```

将 Hugging Face 模型仓库的完整内容放入 `checkpoints/llm/base_model/`。目录中应包含
模型配置、tokenizer 配置及分片权重。也可以通过 `L2R_BASE_MODEL` 指向其他本地目录。

## 训练

在项目根目录运行：

```bash
python train.py
```

训练会：

1. 加载 CTD 数据并创建温跃层跟踪环境。
2. 加载预训练 Attention DQN 作为初始化和 KL 参考策略。
3. 加载冻结的 LLM LoRA 与 Reprogram-TSF 回归模块。
4. 检查 RL 与 LLM 数据使用范围。
5. 执行序列经验回放和 Double-DQN 更新。
6. 写入 TensorBoard 与 SwanLab 日志。
7. 将最终策略保存到 `modelsave/`。

默认训练参数：

| 参数 | 默认值 |
|---|---:|
| 深度范围 | `[-200, 0] m` |
| belief 窗口 | 20 步 |
| 每步温度/盐度采样点 | 各 15 |
| 最大 episode 长度 | 300 |
| 总回合数 | 5000 |
| replay 预热回合 | 100 |
| DQN hidden size | 64 |
| Attention heads / layers | 4 / 2 |

超参数集中在 `config.py` 和 `train.py` 顶部。没有可用 GPU 时，训练入口自动缩短为
50 回合，用于基本调试。

## 配置方式

支持的环境变量：

| 变量 | 用途 | 默认值 |
|---|---|---|
| `L2R_CTD_DIR` | CTD 数据目录 | `data/ctd` |
| `L2R_SPLIT_MANIFEST` | 数据划分清单 | `<CTD_DIR>/split_manifest.json` |
| `L2R_TSF_DATA` | 可选的 TSF 数据集目录 | `data/thermo_tsf` |
| `L2R_PRETRAIN_MODEL` | 预训练 Attention DQN | `checkpoints/pretrained_attention_dqn.pth` |
| `L2R_LORA_DIR` | LLM LoRA adapter | `checkpoints/reprogram/lora_best` |
| `L2R_HEAD_PATH` | Reprogram-TSF 回归模块 | `checkpoints/reprogram/head_best.pt` |
| `L2R_BASE_MODEL` | Hugging Face 基座模型本地目录 | `checkpoints/llm/base_model` |

`.env.example` 仅作为配置模板；项目不会自动读取 `.env`。请在 shell、任务调度器或
容器环境中设置这些变量。

## 版本控制

`.gitignore` 已排除：

- CTD 数据与缓存
- DQN、LoRA、回归头和 ONNX 权重
- TensorBoard、SwanLab 和运行日志
- 图片、输出、虚拟环境及 IDE 配置

首次发布前，请确认所使用的 CTD 数据、Qwen 基座模型和训练权重具有允许公开或再分发
的许可证。项目代码许可证尚未指定，公开前需要由项目所有者选择。

## 当前状态

完整训练需要用户提供 CTD 数据、预训练 Attention DQN、LLM LoRA 和 Reprogram-TSF
回归头。

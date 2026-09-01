# Forecast-to-Reward (F2R) Policy Learning

This repository contains the reinforcement-learning component of **Forecast-to-Reward (F2R)**, an offline forecast-guided framework for lightweight adaptive thermocline sampling with autonomous underwater vehicles (AUVs).

F2R uses an LLM-based thermocline forecaster only during offline policy training. Forecast thermocline-center depths are converted into shaped rewards for learning a residual adaptation head on top of a pretrained attention-based reference policy. After training, the forecaster is removed. The deployed policy maps onboard conductivity-temperature-depth (CTD) observation histories directly to vertical actions and therefore does not require an LLM, an explicit environmental model, or online planning aboard the vehicle.

The companion forecasting implementation, **ThermoTSF-Reprogram**, is available in [ThermoCast](https://github.com/tianzhuoer/ThermoCast).

## Method overview

F2R treats adaptive thermocline sampling as a partially observable sequential decision-making problem. At each step, the AUV has access only to local CTD measurements collected along its trajectory. Its actions determine both where it moves and which part of the water column it observes next.

The framework contains three coordinated components:

1. **Attention-based reference policy.** A pretrained policy, denoted by `Q_ref`, provides basic thermocline-observation behavior, including vertical exploration, repeated observation of high-gradient regions, and compliance with motion boundaries.
2. **ThermoTSF-Reprogram forecaster.** A separately trained and frozen LLM-based forecaster predicts the next five thermocline-center depths from trajectory-dependent depth, temperature, and salinity observations.
3. **Forecast-to-reward adaptation.** Forecasts are converted into dense and vertical-leg-level rewards. Only a residual Q-adapter is optimized; the inherited reference-policy branch remains fixed.

```text
Q_F2R(s, a) = Q_ref(s, a) + DeltaQ_phi(s, a)
```

`DeltaQ_phi` is a two-layer residual Q-adapter whose final layer is initialized to zero. Training combines a temporal-difference objective with an annealed KL regularizer that anchors the adapted policy to the fixed reference policy during early learning.

```text
Partial CTD histories
        |
        +-----------------------> fixed reference branch Q_ref ---+
        |                                                         |
        +-----------------------> residual Q-adapter DeltaQ_phi --+--> vertical action
        |
        +--> frozen ThermoTSF-Reprogram --> forecast depths
                                              |
                                              +--> shaped reward (offline only)

Deployment: CTD histories --> Q_ref + DeltaQ_phi --> vertical action
            (ThermoTSF-Reprogram is removed)
```

## Forecast-informed reward

During offline learning, the frozen forecaster periodically predicts five future thermocline-center depths. Their screened, smoothed mean defines the forecast depth used by the reward. The implementation combines:

- a **dense attraction reward** based on distance to the forecast depth;
- an **exploration reward** for newly visited depth intervals;
- a **leg-span reward** that discourages short-cycle reversals;
- a **crossing reward** for overlap between a completed vertical leg and the forecast thermocline band;
- a **leg-midpoint reward** that encourages sampling on both sides of the forecast depth; and
- a **boundary penalty** for invalid or incomplete vertical steps.

A **vertical leg** is the interval between two successive direction reversals. Leg-level rewards are evaluated only at a turning point and after a minimum leg length, limiting reward exploitation through rapid switching.

## Repository structure

```text
.
|-- train.py                    # F2R policy-training entry point
|-- config.py                   # environment, RL, paths, and logging settings
|-- thermocline_env.py          # Gymnasium thermocline-sampling environment
|-- generate_split_manifest.py  # season-balanced, file-level partitioning
|-- models/
|   `-- attention_dqn.py        # attention-based DQN components
|-- utils/
|   |-- ctd_data.py             # CTD loading and sampling
|   `-- replay_memory.py        # sequential experience replay
|-- llm_tsf/
|   |-- configs/                # frozen forecaster configuration
|   `-- models/reprogram_backbone.py
|-- data/                       # local data; not versioned
|-- checkpoints/                # reference-policy and forecaster weights
`-- modelsave/                  # trained F2R policies
```

## Installation

The code was developed with Python 3.11.13, PyTorch 2.8.0+cu128, and CUDA 12.8.

```bash
conda env create -f environment.yml
conda activate l2r
```

Alternatively:

```bash
pip install -r requirements.txt
```

If CUDA 12.8 is unavailable, install the PyTorch build appropriate for the target driver and runtime before installing the remaining dependencies.

## Data preparation

The environment expects CTD MATLAB files containing a depth grid, temperature profiles, salinity profiles, and a MATLAB `datenum` time grid. The default location is `data/ctd`.

```bash
export L2R_CTD_DIR=/path/to/CTD
export L2R_SPLIT_MANIFEST=/path/to/CTD/split_manifest.json
python generate_split_manifest.py
```

PowerShell:

```powershell
$env:L2R_CTD_DIR = "D:\datasets\CTD"
$env:L2R_SPLIT_MANIFEST = "D:\datasets\CTD\split_manifest.json"
python generate_split_manifest.py
```

The manifest assigns complete files to `rl_train`, `llm_train`, `llm_val`, `llm_test`, `validation`, or `test`. This prevents CTD files used to fine-tune ThermoTSF-Reprogram from being reused to construct F2R policy-training environments.

## Required checkpoints

```text
checkpoints/
|-- pretrained_attention_dqn.pth
|-- llm/
|   `-- base_model/             # complete local Qwen3.5-2B directory
`-- reprogram/
    |-- lora_best/              # LoRA adapter
    `-- head_best.pt            # reprogramming modules and regression head
```

| Variable | Purpose | Default |
|---|---|---|
| `L2R_CTD_DIR` | CTD data directory | `data/ctd` |
| `L2R_SPLIT_MANIFEST` | file-level split manifest | `<CTD_DIR>/split_manifest.json` |
| `L2R_TSF_DATA` | optional prepared forecasting dataset | `data/thermo_tsf` |
| `L2R_PRETRAIN_MODEL` | pretrained reference policy | `checkpoints/pretrained_attention_dqn.pth` |
| `L2R_LORA_DIR` | ThermoTSF-Reprogram LoRA adapter | `checkpoints/reprogram/lora_best` |
| `L2R_HEAD_PATH` | reprogramming modules and regression head | `checkpoints/reprogram/head_best.pt` |
| `L2R_BASE_MODEL` | local Qwen3.5-2B directory | `checkpoints/llm/base_model` |

The repository does not automatically load `.env`; set variables in the shell, scheduler, or container environment.

## Training

```bash
python train.py
```

The pipeline loads the CTD environments, initializes the fixed reference branch, loads the frozen forecaster, constructs forecast-informed rewards, and updates only the residual Q-adapter using sequential experience replay, temporal-difference learning, and annealed KL regularization. Logs are written to TensorBoard and SwanLab, and trained policies are saved under `modelsave/`.

| Setting | Default |
|---|---:|
| Policy depth range | `[-200, 0] m` |
| CTD belief window | 20 steps |
| Partial temperature/salinity profile | 15 bins per modality |
| Vertical action increment | 5 m |
| Maximum episode length | 300 steps |
| Attention hidden size | 64 |
| Attention heads / decoder layers | 4 / 2 |
| Initial / final KL coefficient | 5.0 / 0.3 |
| KL annealing duration | 1,000 episodes |

## Scope and reproducibility

- The forecaster supplies information only through offline reward shaping and is not part of onboard inference.
- The policy controls vertical motion only and represents the thermocline by a single maximum-gradient center depth.
- The manuscript experiments use South China Sea CTD records; the results should not be interpreted as global validation.
- Data, pretrained models, checkpoints, and experiment logs are not distributed here.
- This repository currently has no explicit software license. Obtain permission before reuse and verify the licenses of all datasets and pretrained weights.

## Citation

To be updated.





import os
import random


DQN_HIDDEN_SIZE = 64
DQN_BATCH_SIZE  = 32
SEQ_LEN         = DQN_BATCH_SIZE
NUM_HEADS       = 4
NUM_LAYERS      = 2


DQN_GAMMA     = 0.996
DQN_EPS_START = 0.99
DQN_FT_EPS_START=0.45
DQN_EPS_END   = 0.02
DQN_EPS_DECAY = 600000
DQN_TAU       = 0.005
DQN_LR        = 2e-5
DQN_FT_LR=1e-5


REPLAY_MEMORY_SIZE = 100
LOG_INTERVAL       = 10


IS_TRAIN           = True
MODEL_TYPE         = 'attention'   # 'rnn' or 'attention'
MANUAL_SEED        = random.randint(0, 999999)
RENDER_MODE        = None
TRAINING_EPISODES  = 5000 - REPLAY_MEMORY_SIZE


ENV_ID              = "Thermocline_simple/Thermocline_track-v1"
MAX_EPISODES_LEN    = 300
BELIEF_WINDOW_SIZE  = 20
SAMPLE_WINDOW_SIZE  = 15
DEPTH_MIN           = -200
DEPTH_MAX           = 0
GRA_THRESHOLD       = 1
STEP_SIZE           = 5
PLOT_GRD_PERCENT    = 90
SAMPLE_DAYS         = 20


GRA_COUNT = 15
R_GLOBAL  = 400
R_DIR1    = 5
R_DIR2    = 5
R_GRAD    = 5

R_LLM_DENSE_MAX  = 10.0
R_LLM_SIGMA      = 50.0


LLM_DIST_THRESH  = 10.0
R_LLM_APPROACH   = 2.0
R_LLM_RECEDE     = 1.0


PRIOR_MODEL_PATH = "modelsave/final/myDQNmodelAttn202601230418.pth"
KL_WEIGHT        = 0.3
KL_TEMPERATURE   = 1.0


REPLAY_START = REPLAY_MEMORY_SIZE * MAX_EPISODES_LEN


CTD_FOLDER_PATH      = os.environ.get("L2R_CTD_DIR", "data/ctd")
LOG_DIR_TENSORBOARD  = "tensorboard_log/thermo_DQN"
LOG_DIR_MODELSAVE    = "modelsave/myDQNmodel"
PRETRAIN_MODEL_PATH  = os.environ.get(
    "L2R_PRETRAIN_MODEL",
    "checkpoints/pretrained_attention_dqn.pth",
)
LLM_LORA_DIR = os.environ.get(
    "L2R_LORA_DIR",
    "checkpoints/reprogram/lora_best",
)
LLM_HEAD_PATH = os.environ.get(
    "L2R_HEAD_PATH",
    "checkpoints/reprogram/head_best.pt",
)







LABEL_THRESHOLD    = 0.21
LABEL_THRESHOLD_NOISE = 0.002


TSF_K = 20
TSF_H = 5


CUDA_DEVICE_DQN = "1"
CUDA_DEVICE_LLM = "1"


MAX_LENGTH    = 1024
PROMPT        = "实验的目标是让AUV自主找到温度梯度最大的深度层并在此附近往复采样。根据以下给出的数据片段，分析AUV的采样行为，判断当前AUV工作在何种阶段,用于指导下一步的AUV行为。通常前100步为探索早期，侧重探索，后期则侧重温往复采样。"
CKPT_INTERVAL = 30
CLS_HEAD_PATH  = "MissionFT/checkpoints/cls_head_best.pt"


SWANLAB_PROJECT_LLM_DQN    = "ThermoRL-LLM-DQN"
SWANLAB_EXPERIMENT_LLM_DQN = "llm-dqn-attn"


SWANLAB_PROJECT_PGD    = "ThermoRL-PGD-DQN"
SWANLAB_EXPERIMENT_PGD = "pgd-dqn-attn"


CTD_SPLIT_MANIFEST = os.environ.get(
    "L2R_SPLIT_MANIFEST",
    os.path.join(CTD_FOLDER_PATH, "split_manifest.json"),
)


CTD_YEAR_BLOCK_SIZE = 3
CTD_YEAR_WM_COUNT   = 2

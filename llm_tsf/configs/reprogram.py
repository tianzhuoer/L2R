from .shared import *


MODEL_PATH = os.environ.get("L2R_BASE_MODEL", "checkpoints/llm/base_model")


REPROGRAM_D_MODEL      = 64
REPROGRAM_N_HEADS      = 4
REPROGRAM_N_PROTOTYPES = 128
REPROGRAM_DROPOUT      = 0.1


BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 8


LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGET  = ["k_proj", "q_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]



LR_LORA      = 1e-5
LR_REPROGRAM = 8e-5
LR_HEAD      = 8e-5
WARMUP_RATIO = 0.08
GRAD_CLIP_MAX_NORM = 0.5


DEPTH_NOISE_STD = 1.5
FEAT_NOISE_STD  = 0.02




TSF_CKPT_ROOT = "checkpoints/reprogram"
TSF_CKPT_DIR  = TSF_CKPT_ROOT
TSF_LORA_BEST = f"{TSF_CKPT_DIR}/lora_best"
TSF_HEAD_BEST = f"{TSF_CKPT_DIR}/head_best.pt"

# ── SwanLab ────────────────────────────────────────────────────────────────────
SWANLAB_EXPERIMENT = f"qwen3.5-2b-reprogram-tsf-{MAX_SAMPLES_PER_EPOCH//1000}k"

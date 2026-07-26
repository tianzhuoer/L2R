import os


# MANUAL_SEED = random.randint(0, 999999)
MANUAL_SEED = 20058


CTD_FOLDER_PATH    = os.environ.get("L2R_CTD_DIR", "data/ctd")
DEPTH_MIN          = -250
DEPTH_MAX          = 0
SAMPLE_WINDOW_SIZE = 15
STEP_SIZE          = 5


TSF_K = 32
TSF_H = 5



N_INPUT_FEATURES = 8



TSF_DEPTH_MIN = -150
TSF_DEPTH_MAX = -25



SPLIT_TRAIN_MOD = set(range(7))
SPLIT_VAL_MOD   = {7, 8}
SPLIT_TEST_MOD  = {9}


SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
SEASON_TYPICAL_DEPTHS = {
    0: "65–125 m",
    1: "45–105 m",
    2: "25–85 m",
    3: "45–125 m",
}


CTD_TSF_AUV_NOISE_STD    = 7.0
CTD_TSF_NEAR_THERMO_PROB = 0.5
CTD_TSF_STRIDE           = 5
CTD_TSF_N_AUGMENT        = 12
MAX_SAMPLES_PER_EPOCH    = 16800
CTD_AUV_EXCLUDE_KEYWORDS = ('AUV_',)
CTD_YEAR_BLOCK_SIZE      = 3
CTD_YEAR_WM_COUNT        = 2


CTD_SPLIT_MANIFEST = os.environ.get(
    "L2R_SPLIT_MANIFEST",
    os.path.join(CTD_FOLDER_PATH, "split_manifest.json"),
)



TSF_DATA_PATH = os.environ.get("L2R_TSF_DATA", "data/thermo_tsf")




DATASET_CACHE_DIR = "dataset_cache"


EPOCHS           = 100
GRAD_ACCUM_STEPS = 4
ES_PATIENCE      = 10
ES_MIN_EPOCHS    = 50
CKPT_INTERVAL    = 10
WEIGHT_DECAY     = 1e-3
HUBER_DELTA      = 0.2

# ── GPU ────────────────────────────────────────────────────────────────────────
CUDA_DEVICE_LLM = "1"
CUDA_DEVICE_DQN = "0"


SWANLAB_PROJECT    = "ThermoTSF-Reprogram"
SWANLAB_LOG_INTERVAL = 50

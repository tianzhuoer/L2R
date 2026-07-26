

import os
import sys
import time
import math
import random
from itertools import count

import swanlab

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        """No-op fallback for inference/evaluation environments without TensorBoard."""
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

import gymnasium as gym
from thermocline_env import ThermoTrackEnv  # noqa: F401


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TSF_DIR    = os.path.join(_SCRIPT_DIR, "llm_tsf")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _TSF_DIR not in sys.path:
    sys.path.append(_TSF_DIR)


from config import (
    DQN_HIDDEN_SIZE, DQN_BATCH_SIZE, SEQ_LEN, DQN_GAMMA,
    DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY, DQN_TAU,
    REPLAY_MEMORY_SIZE, LOG_INTERVAL, TRAINING_EPISODES,
    ENV_ID, RENDER_MODE, BELIEF_WINDOW_SIZE, SAMPLE_WINDOW_SIZE,
    DEPTH_MIN, DEPTH_MAX, NUM_HEADS, NUM_LAYERS,
    LOG_DIR_TENSORBOARD, LOG_DIR_MODELSAVE, TSF_H,
    SWANLAB_PROJECT_LLM_DQN, SWANLAB_EXPERIMENT_LLM_DQN, DQN_FT_EPS_START,
    DQN_FT_LR,
    CUDA_DEVICE_DQN, MANUAL_SEED, MAX_EPISODES_LEN,
)
import models.attention_dqn as attn_models
from utils.replay_memory import SeqReplayMemory


KL_POLICY_TAU      = 1.0




#         w_rl(t) = START + (FLOOR - START) · t




KL_LAMBDA_RL_START  = 5.0
KL_LAMBDA_RL_FLOOR  = 0.3
KL_ANNEAL_EPS       = 1000






LLM_BONUS            = 40.0
LLM_BONUS_HALF_WIDTH = 10.0
LLM_MID_BONUS        = 60.0
LLM_MID_DIST_MAX     = 30.0
LLM_INFER_INTERVAL   = 20
LLM_MIN_LEG_LEN      = 5



LEG_SPAN_BONUS_GAIN  = 0.6
LEG_SPAN_CAP         = 80.0



LLM_DEPTH_TRUST_MAX  = -100.0





EXPLORE_BIN_BONUS    = 8.0
EXPLORE_N_BINS       = 25
EXPLORE_SHALLOW_GAIN = 1.0



def _explore_bin_bonus(depth: float, visited_bins: set) -> float:
    
    span = DEPTH_MAX - DEPTH_MIN
    if span <= 0:
        return 0.0
    frac = (depth - DEPTH_MIN) / span
    frac = min(max(frac, 0.0), 1.0)
    b = int(frac * EXPLORE_N_BINS)
    if b >= EXPLORE_N_BINS:
        b = EXPLORE_N_BINS - 1
    if b in visited_bins:
        return 0.0
    visited_bins.add(b)
    shallow_frac = b / max(EXPLORE_N_BINS - 1, 1)
    return EXPLORE_BIN_BONUS * (1.0 + EXPLORE_SHALLOW_GAIN * shallow_frac)




_CKPT_SUBDIR  = os.path.join("checkpoints", "reprogram")
_LORA16K_DIR  = os.path.join(_TSF_DIR, _CKPT_SUBDIR, "lora-16k")
if os.path.isdir(_LORA16K_DIR):
    _DEFAULT_LORA = os.path.join(_LORA16K_DIR, "lora_best")
    _DEFAULT_HEAD = os.path.join(_LORA16K_DIR, "head_best.pt")
else:
    _DEFAULT_LORA = os.path.join(_TSF_DIR, _CKPT_SUBDIR, "lora_best")
    _DEFAULT_HEAD = os.path.join(_TSF_DIR, _CKPT_SUBDIR, "head_best.pt")


_BELIEF_STEPS = BELIEF_WINDOW_SIZE
_OBS_DIM      = SAMPLE_WINDOW_SIZE * 2 + 1
_FLAT_DIM     = _BELIEF_STEPS * _OBS_DIM


_TSF_DEPTH_MIN = -150.0
_TSF_DEPTH_MAX = -25.0
_TSF_D_RANGE   = _TSF_DEPTH_MAX - _TSF_DEPTH_MIN


def _norm_depth(z: float) -> float:
    return (z - _TSF_DEPTH_MIN) / _TSF_D_RANGE * 2.0 - 1.0






def check_data_overlap(ctd_folder: str = None, verbose: bool = True) -> dict:
    
    import datetime
    import scipy.io

    try:
        import importlib
        _shared = importlib.import_module("configs.shared")
        _DEFAULT_CTD         = _shared.CTD_FOLDER_PATH
        CTD_AUV_EXCLUDE_KEYWORDS = _shared.CTD_AUV_EXCLUDE_KEYWORDS
        CTD_YEAR_BLOCK_SIZE  = _shared.CTD_YEAR_BLOCK_SIZE
        CTD_YEAR_WM_COUNT    = _shared.CTD_YEAR_WM_COUNT
    except Exception:
        print("[check_data_overlap] 无法导入 LLM_ThermoTSF 配置，跳过检查。")
        return {}

    def _dn2dt(dn):
        return (datetime.datetime.fromordinal(int(dn))
                + datetime.timedelta(days=dn % 1)
                - datetime.timedelta(days=366))

    folder  = ctd_folder or _DEFAULT_CTD
    results = {}
    all_files = sorted(f for f in os.listdir(folder) if f.endswith('.mat'))
    mat_files = [f for f in all_files
                 if not any(kw in f for kw in CTD_AUV_EXCLUDE_KEYWORDS)]

    for fname in mat_files:
        try:
            data = scipy.io.loadmat(os.path.join(folder, fname), variable_names=['new_time_grid'])
            tg   = np.asarray(data['new_time_grid']).flatten()
        except Exception:
            continue

        N       = len(tg)
        n_train = int(N * 0.70)
        n_val   = int(N * 0.20)

        try:
            dts      = [_dn2dt(d) for d in tg]
            years    = sorted({dt.year for dt in dts})
            wm_years = {y for i, y in enumerate(years)
                        if i % CTD_YEAR_BLOCK_SIZE < CTD_YEAR_WM_COUNT}
            train_end_date  = dts[n_train - 1]
            test_start_date = dts[n_train + n_val]
        except Exception:
            continue

        results[fname] = {
            'n_total':         N,
            'n_train':         n_train,
            'train_end_date':  train_end_date,
            'test_start_date': test_start_date,
            'wm_years':        sorted(wm_years),
        }

        if verbose:
            print(f"  {fname}: "
                  f"LLM-train ≤ {train_end_date.strftime('%Y-%m')}  "
                  f"RL-safe   ≥ {test_start_date.strftime('%Y-%m')}  "
                  f"wm_years={sorted(wm_years)}")

    if results and verbose:
        print(
            "\n[check_data_overlap] 建议：\n"
            "  RL 环境应只采样每个文件中 test_start_date 之后的数据段（最后 10%），\n"
            "  以避免与 LLM 训练数据（前 70%）重叠。\n"
            "  在 RL env 初始化时传入 split='test' 或 start_ratio=0.9 等参数。"
        )

    return results






class AttnDQNPolicy(nn.Module):
    

    def __init__(self, n_observations: int, n_actions: int, H: int = 5,
                 num_hiddens: int = 64, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1,
                 depth_range=(-250, 0), depth_feature_idx: int = 0):
        super().__init__()
        self.H = H

        self.state_embedding = attn_models.StateEmbedding(
            total_dim=n_observations,
            depth_dim=1,
            belief_len=_BELIEF_STEPS,
            embed_dim=num_hiddens,
        )
        self.positional_encoding = attn_models.PositionalEncoding(num_hiddens, dropout)
        self.decoder = attn_models.Decoder(
            tgt_size=n_observations,
            key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens,
            num_hiddens=num_hiddens, norm_shape=num_hiddens,
            ffn_num_input=num_hiddens, ffn_num_hiddens=num_hiddens * 2,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        )

        self.output_projection = nn.Linear(num_hiddens, n_actions)



        self.q_adapter = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.GELU(),
            nn.Linear(num_hiddens, n_actions),
        )
        nn.init.zeros_(self.q_adapter[-1].weight)
        nn.init.zeros_(self.q_adapter[-1].bias)

        self.aux_head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, H),
        )

    def freeze_backbone(self):
        
        for p in self.state_embedding.parameters():
            p.requires_grad_(False)
        for p in self.positional_encoding.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        for p in self.output_projection.parameters():
            p.requires_grad_(False)
        for p in self.aux_head.parameters():
            p.requires_grad_(False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_proj    = self.state_embedding(x)
        x_encoded = self.positional_encoding(x_proj)
        return self.decoder(x_encoded, attn_mask=None)

    def forward(self, x, data_mask=None, attn_mask=None, use_adapter=True):
        encoded  = self._encode(x)

        q_values = self.output_projection(encoded)
        if use_adapter:
            q_values = q_values + self.q_adapter(encoded)
        return q_values.squeeze(1)

    def forward_with_aux(self, x, data_mask=None, attn_mask=None):
        encoded     = self._encode(x)
        q_values    = self.output_projection(encoded) + self.q_adapter(encoded)
        last_hidden = encoded[:, -1, :]
        aux_pred    = self.aux_head(last_hidden)
        return q_values.squeeze(1), aux_pred






class LLMDepthPredictor:
    

    def __init__(self, lora_dir: str, head_path: str, device, H: int = 5):
        self.model    = None
        self._forecaster = None
        self.device   = device
        self.H        = H
        self._load(lora_dir, head_path)



    def _load(self, lora_dir: str, head_path: str):
        if not (os.path.isdir(lora_dir) and os.path.isfile(head_path)):
            print(f"[LLMDepthPredictor] 检查点不存在，LLM 奖励塑形已禁用。\n"
                  f"  lora_dir : {lora_dir}\n  head_path: {head_path}")
            return
        try:
            import importlib
            import importlib.util as _ilu


            rp_cfg = importlib.import_module("configs.reprogram")


            _rb_path = os.path.join(_TSF_DIR, "models", "reprogram_backbone.py")
            _rb_spec = _ilu.spec_from_file_location("reprogram_backbone", _rb_path)
            _rb_mod  = _ilu.module_from_spec(_rb_spec)
            _rb_spec.loader.exec_module(_rb_mod)
            ThermoForecasterReprogram = _rb_mod.ThermoForecasterReprogram

            from transformers import AutoModelForCausalLM


            _adapter_cfg_path = os.path.join(lora_dir, "adapter_config.json")
            if not os.path.isfile(_adapter_cfg_path):
                import json as _json
                _adapter_cfg = {
                    "base_model_name_or_path": rp_cfg.MODEL_PATH,
                    "bias": "none",
                    "fan_in_fan_out": False,
                    "inference_mode": True,
                    "init_lora_weights": True,
                    "lora_alpha": rp_cfg.LORA_ALPHA,
                    "lora_dropout": rp_cfg.LORA_DROPOUT,
                    "modules_to_save": None,
                    "peft_type": "LORA",
                    "r": rp_cfg.LORA_R,
                    "target_modules": rp_cfg.LORA_TARGET,
                    "task_type": "CAUSAL_LM",
                }
                with open(_adapter_cfg_path, "w") as _f:
                    _json.dump(_adapter_cfg, _f, indent=2)
                print(f"[LLMDepthPredictor] adapter_config.json 已重建 "
                      f"(r={rp_cfg.LORA_R}, target={rp_cfg.LORA_TARGET})")


            import pathlib as _pl
            _model_path = _pl.Path(rp_cfg.MODEL_PATH).as_posix()
            print(f"[LLMDepthPredictor] 加载基座模型 {_model_path} …")
            base = AutoModelForCausalLM.from_pretrained(
                _model_path,
                device_map=None,
                dtype=torch.bfloat16,
            )

            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(
                _model_path, trust_remote_code=True, local_files_only=True
            )
            print(f"[LLMDepthPredictor] 加载 LoRA adapter {lora_dir} …")
            from peft import get_peft_model, LoraConfig, TaskType
            from safetensors.torch import load_file as _load_safetensors
            _lora_cfg = LoraConfig(
                r=rp_cfg.LORA_R,
                lora_alpha=rp_cfg.LORA_ALPHA,
                target_modules=rp_cfg.LORA_TARGET,
                lora_dropout=rp_cfg.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
            )
            peft_model = get_peft_model(base, _lora_cfg)
            _st_path = os.path.join(lora_dir, "model.safetensors")
            _sd = _load_safetensors(_st_path, device="cpu")
            _missing, _unexpected = peft_model.load_state_dict(_sd, strict=False)
            if _unexpected:
                print(f"[LLMDepthPredictor] LoRA 权重未使用键（忽略）: {_unexpected[:5]}")
            peft_model.eval()


            forecaster = ThermoForecasterReprogram(
                peft_model   = peft_model,
                tokenizer    = _tokenizer,
                H            = self.H,
                K            = _BELIEF_STEPS,
                N            = SAMPLE_WINDOW_SIZE,
                d_model      = rp_cfg.REPROGRAM_D_MODEL,
                n_heads      = rp_cfg.REPROGRAM_N_HEADS,
                n_prototypes = rp_cfg.REPROGRAM_N_PROTOTYPES,
                dropout      = rp_cfg.REPROGRAM_DROPOUT,
            )
            ckpt = torch.load(head_path, map_location="cpu")

            if isinstance(ckpt, dict) and "reprogram" in ckpt and len(ckpt) == 1:
                ckpt = ckpt["reprogram"]


            _h_missing, _h_unexpected = forecaster.load_state_dict(ckpt, strict=False)
            if _h_unexpected:
                print(f"[LLMDepthPredictor] head 权重未使用键（忽略）: {_h_unexpected[:3]}")

            forecaster = forecaster.to(self.device).eval()
            for p in forecaster.parameters():
                p.requires_grad_(False)

            self._forecaster = forecaster
            self.model    = True
            print("[LLMDepthPredictor] 模型加载成功，参数全部冻结。")
        except Exception as e:
            print(f"[LLMDepthPredictor] 加载失败（LLM 奖励塑形已禁用）：{e}")
            self.model = None



    def infer(self, state_batch: torch.Tensor,
              episode_datetime=None) -> torch.Tensor | None:
        
        if self.model is None:
            return None


        last = state_batch[:, -1, :].float()                        # (B, 620)
        obs  = last.reshape(-1, _BELIEF_STEPS, _OBS_DIM)            # (B, 20, 31)

        depths  = obs[:, :, 0:1]                                              # (B, 20, 1)
        T_profs = obs[:, :, 1:1+SAMPLE_WINDOW_SIZE]                           # (B, 20, 15)
        S_profs = obs[:, :, 1+SAMPLE_WINDOW_SIZE:1+SAMPLE_WINDOW_SIZE*2]      # (B, 20, 15)


        # context = [season_norm, doy_sin, doy_cos, depth_mean_n, depth_range_n,
        #            T_mean_n, T_std_n, S_mean_n, S_std_n]  → CTX_DIM=9
        season_id = 0
        try:
            import math as _m

            if episode_datetime is not None:
                doy = episode_datetime.timetuple().tm_yday
                month = episode_datetime.month
                season_id = (month % 12) // 3
                season_norm = season_id / 3.0
                doy_sin = _m.sin(2 * _m.pi * doy / 365.25)
                doy_cos = _m.cos(2 * _m.pi * doy / 365.25)
            else:
                season_norm = doy_sin = doy_cos = 0.0


            depths_arr = depths[:, :, 0].cpu().float().numpy()         # (B, K)
            depth_mean_vals = depths_arr.mean(axis=1)                  # (B,)
            depth_mean_n  = (depth_mean_vals - _TSF_DEPTH_MIN) / _TSF_D_RANGE * 2.0 - 1.0
            depth_range_n = (depths_arr.max(axis=1) - depths_arr.min(axis=1)) / (-_TSF_DEPTH_MIN) * 2.0 - 1.0


            T_np = T_profs.cpu().float().numpy()                       # (B, K, N)
            T_mu  = T_np.mean(axis=(1, 2))
            T_sig = T_np.std(axis=(1, 2)) + 1e-6
            T_mean_n = (T_mu / T_sig).clip(-3, 3) / 3.0
            T_std_n  = (T_sig / (T_mu + 1e-6 + 10.0)).clip(-3, 3) / 3.0


            S_mean_n = np.zeros(obs.shape[0])
            S_std_n  = np.zeros(obs.shape[0])

            ctx_np = np.stack([
                np.full(obs.shape[0], season_norm),
                np.full(obs.shape[0], doy_sin),
                np.full(obs.shape[0], doy_cos),
                depth_mean_n, depth_range_n,
                T_mean_n, T_std_n, S_mean_n, S_std_n,
            ], axis=1).astype(np.float32)                              # (B, 9)
            ctx = torch.from_numpy(ctx_np).to(self.device)
        except Exception:
            ctx = torch.zeros(obs.shape[0], 9, device=self.device)

        depths  = depths.to(self.device)
        T_profs = T_profs.to(self.device)

        season_ids = torch.full((obs.shape[0],), int(season_id),
                                dtype=torch.long, device=self.device)

        with torch.no_grad():
            preds = self._forecaster(depths, T_profs, S_profs, ctx, season_ids)  # (B, H)
        return preds  # (B, H)，[-1, 1]






class LLMRewardShapeDQNAgent:
    

    def __init__(self, env, seed=None, use_cuda: bool = True,
                 log_dir: str = None, model_path: str = None,
                 kl_policy_tau:           float = KL_POLICY_TAU,
                 kl_lambda_rl_start:      float = KL_LAMBDA_RL_START,
                 kl_lambda_rl_floor:      float = KL_LAMBDA_RL_FLOOR,
                 kl_anneal_eps:           int   = KL_ANNEAL_EPS,
                 llm_bonus:               float = LLM_BONUS,
                 llm_bonus_half_width:    float = LLM_BONUS_HALF_WIDTH,
                 llm_infer_interval:      int   = LLM_INFER_INTERVAL,
                 lora_dir: str = None, head_path: str = None,
                 finetune_lr:             float = None,
                 finetune_eps_start:      float = None):
        self.env                   = env
        self.kl_policy_tau           = kl_policy_tau
        self.kl_lambda_rl_start      = kl_lambda_rl_start
        self.kl_lambda_rl_floor      = kl_lambda_rl_floor
        self.kl_anneal_eps           = kl_anneal_eps
        self.llm_bonus               = llm_bonus
        self.llm_bonus_half_width    = llm_bonus_half_width
        self.llm_infer_interval      = llm_infer_interval
        self._i_ep                   = 0
        self._opt_step               = 0

        self._leg_states: list       = []   # list of (state, action, next_state, r, cm, nm)
        self._leg_depths: list       = []
        self._leg_action_prev: int   = -1


        self.BATCH_SIZE = DQN_BATCH_SIZE
        self.SEQ_LEN    = SEQ_LEN
        self.GAMMA      = DQN_GAMMA
        self.EPS_START  = finetune_eps_start if finetune_eps_start is not None else DQN_EPS_START
        self.EPS_END    = DQN_EPS_END
        self.EPS_DECAY  = DQN_EPS_DECAY
        self.TAU        = DQN_TAU
        self.LR         = finetune_lr if finetune_lr is not None else DQN_FT_LR


        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            self.env.action_space.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)


        self.device = torch.device(
            f"cuda:{CUDA_DEVICE_DQN}" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        print(f"[Agent] 使用设备: {self.device}")


        self.n_actions = env.action_space.n
        state, _ = env.reset()
        self.n_observations = state.shape[-1]  # 31 (depth + T×15 + S×15)


        _net_kw = dict(
            n_observations=self.n_observations,
            n_actions=self.n_actions,
            H=TSF_H,
            num_hiddens=DQN_HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=0.1,
            depth_range=(DEPTH_MIN, DEPTH_MAX),
            depth_feature_idx=0,
        )
        self.policy_net = AttnDQNPolicy(**_net_kw).to(self.device)
        self.target_net = AttnDQNPolicy(**_net_kw).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())


        self.ref_net = AttnDQNPolicy(**_net_kw).to(self.device)


        if model_path and os.path.isfile(model_path):
            ckpt = torch.load(model_path, map_location=self.device)

            ckpt = {k: v for k, v in ckpt.items() if not k.startswith("q_adapter")}
            missing, unexpected = self.policy_net.load_state_dict(ckpt, strict=False)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.ref_net.load_state_dict(self.policy_net.state_dict())
            if missing:
                print(f"[Agent] 新增参数（随机初始化）: {missing}")
            if unexpected:
                print(f"[Agent] 未使用的旧参数（忽略）: {unexpected}")
            print(f"[Agent] 预训练权重已加载 ← {model_path}")
        self.ref_net.eval()
        for p in self.ref_net.parameters():
            p.requires_grad_(False)


        self.policy_net.freeze_backbone()
        self.target_net.freeze_backbone()
        n_trainable = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print(f"[Agent] 可训练参数：{n_trainable}（仅 q_adapter）")


        _lora = lora_dir  or _DEFAULT_LORA
        _head = head_path or _DEFAULT_HEAD
        self.llm_predictor = LLMDepthPredictor(_lora, _head, device=self.device, H=TSF_H)
        if self.llm_predictor.model is None:
            print("[Agent] LLM 预测器不可用，奖励塑形禁用。")
            self.llm_bonus = 0.0


        self.optimizer     = optim.AdamW(
            self.policy_net.q_adapter.parameters(), lr=self.LR, amsgrad=True
        )
        self.replay_memory = SeqReplayMemory(seq_length=self.SEQ_LEN)

        self.steps_done     = 0
        self.recent_losses   = []
        self.recent_returns  = []
        self.recent_lengths  = []
        self.recent_llm_bias = []
        self._last_llm_depth = float('nan')
        self._llm_depth_ema_alpha = 0.05

        # ── TensorBoard ─────────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=log_dir)
        self._log_hyperparams()



    def _log_hyperparams(self):
        n_trainable = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.policy_net.parameters())
        text = (
            f"backbone=frozen  trainable={n_trainable}/{n_total} params (q_adapter only)\n"
            f"KL(π‖π_ref)  w_rl={self.kl_lambda_rl_start}→{self.kl_lambda_rl_floor}(衰减)  anneal={self.kl_anneal_eps}ep  "
            f"kl_policy_tau={self.kl_policy_tau}\n"
            f"llm_bonus={self.llm_bonus}  half_width={self.llm_bonus_half_width}m  "
            f"min_leg_len={LLM_MIN_LEG_LEN}  infer_interval={self.llm_infer_interval}\n"
            f"LR={self.LR}  GAMMA={self.GAMMA}  TAU={self.TAU}\n"
            f"EPS_START={self.EPS_START}  EPS_END={self.EPS_END}  "
            f"EPS_DECAY={self.EPS_DECAY}\n"
            f"BATCH={self.BATCH_SIZE}  SEQ={self.SEQ_LEN}  "
            f"H={TSF_H}  n_obs={self.n_observations}  "
            f"n_actions={self.n_actions}\n"
            f"LLM predictor loaded: {self.llm_predictor.model is not None}"
        )
        print(text)
        self.writer.add_text("Model/Hyperparameters", text, 0)

    @property
    def _kl_weight(self) -> float:
        
        t = min(self._i_ep / max(self.kl_anneal_eps, 1), 1.0)
        return self.kl_lambda_rl_start + (self.kl_lambda_rl_floor - self.kl_lambda_rl_start) * t



    def select_action(self, state: torch.Tensor,
                      data_mask=None, attn_mask=None) -> torch.Tensor:
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * \
              math.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                q = self.policy_net(state, data_mask, attn_mask)
                return q.max(1).indices.view(1, 1)
        return torch.tensor(
            [[self.env.action_space.sample()]],
            device=self.device, dtype=torch.long,
        )

    def _rl_kl_loss(self, q_values: torch.Tensor,
                    state_batch: torch.Tensor,
                    mask_batch) -> torch.Tensor:
        
        eps = 1e-7
        q_last = q_values[:, -1, :] if q_values.dim() == 3 else q_values  # (B, A)
        pi = F.softmax(q_last / self.kl_policy_tau, dim=-1) + eps          # (B, A)

        with torch.no_grad():
            q_ref = self.ref_net(state_batch, mask_batch, use_adapter=False)
            q_ref_last = q_ref[:, -1, :] if q_ref.dim() == 3 else q_ref   # (B, A)
            p_ref = F.softmax(q_ref_last / self.kl_policy_tau, dim=-1) + eps

        kl = (pi * (pi.log() - p_ref.log())).sum(dim=-1).mean()
        return kl



    def optimize_model(self):
        if len(self.replay_memory) < REPLAY_MEMORY_SIZE:
            return None

        batch = self.replay_memory.sample(self.BATCH_SIZE)

        state_batch      = batch.state       # (B, S, 620)
        action_batch     = batch.action      # (B, S)
        next_state_batch = batch.next_state  # (B, S, 620)
        reward_batch     = batch.reward      # (B, S)
        mask_batch       = batch.mask
        next_mask_batch  = batch.next_mask

        try:
            _ = mask_batch.shape
        except AttributeError:
            mask_batch = next_mask_batch = None


        self._opt_step += 1
        w_rl = self._kl_weight


        q_values = self.policy_net(state_batch, mask_batch)
        # q_values: (B, S, A)


        act3d = action_batch.unsqueeze(2)               # (B, S, 1)
        sav   = q_values.gather(2, act3d).squeeze(2)   # (B, S)

        with torch.no_grad():

            argmax_a = self.policy_net(next_state_batch, next_mask_batch) \
                           .argmax(dim=2, keepdim=True)               # (B, S, 1)

            next_q   = self.target_net(next_state_batch, next_mask_batch,
                                       use_adapter=False)
            next_sv  = next_q.gather(2, argmax_a).squeeze(2)         # (B, S)

        expected = reward_batch + self.GAMMA * next_sv
        td_loss  = F.smooth_l1_loss(sav, expected)


        kl_rl_loss = torch.tensor(0.0, device=self.device)
        if w_rl > 0:
            kl_rl_loss = self._rl_kl_loss(q_values, state_batch, mask_batch)


        loss = td_loss + w_rl * kl_rl_loss

        self.optimizer.zero_grad()
        loss.backward()
        grads = [p for p in self.policy_net.parameters() if p.grad is not None]
        if grads:
            torch.nn.utils.clip_grad_value_(grads, 100.0)
        self.optimizer.step()

        return loss.item(), td_loss.item(), kl_rl_loss.item()



    def _soft_update(self):

        for k, p in self.policy_net.q_adapter.named_parameters():
            t = dict(self.target_net.q_adapter.named_parameters())[k]
            t.data.copy_(p.data * self.TAU + t.data * (1 - self.TAU))



    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        print(f"[Agent] 策略网络已保存 → {path}")

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.policy_net.eval()
        print(f"[Agent] 策略网络已加载 ← {path}")



    def train(self, num_episodes: int = 600, log_interval: int = 50):
        t0 = time.time()

        for i_ep in range(num_episodes):
            state, info = self.env.reset()


            self.env.unwrapped.llm_depth = self._last_llm_depth
            state = np.nan_to_num(state.flatten())
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            cur_mask = info.get("mask", None)

            ep_loss = ep_td = ep_kl_rl = ep_ret = 0.0
            ep_llm_bonus_count = 0
            ep_explore_bonus = 0.0
            visited_bins: set = set()
            ep_actions: list[int] = []
            ep_depths:  list[float] = []


            leg_buf: list = []   # list of (state, action, next_state, r_t, cm_t, nm_t)
            leg_depths: list = []
            prev_action: int = -1


            step_in_ep = 0

            for t in count():
                mask_t = (
                    torch.tensor(
                        cur_mask, dtype=torch.bool, device=self.device
                    ).unsqueeze(0)
                    if cur_mask is not None else None
                )

                cur_depth = state[0, -_OBS_DIM].item()
                ep_depths.append(cur_depth)


                if (self.llm_predictor.model is not None
                        and step_in_ep % self.llm_infer_interval == 0):
                    with torch.no_grad():
                        preds = self.llm_predictor.infer(
                            state.unsqueeze(1),
                            episode_datetime=self.env.unwrapped.episode_start_datetime,
                        )
                    if preds is not None:
                        z_norm = preds.mean(dim=1).mean().item()
                        raw_depth = (z_norm + 1.0) / 2.0 * _TSF_D_RANGE + _TSF_DEPTH_MIN



                        trusted = raw_depth >= LLM_DEPTH_TRUST_MAX
                        if trusted or math.isnan(self._last_llm_depth):
                            use_depth = raw_depth if trusted else LLM_DEPTH_TRUST_MAX
                            if math.isnan(self._last_llm_depth):
                                self._last_llm_depth = use_depth
                            else:
                                self._last_llm_depth = (
                                    self._llm_depth_ema_alpha * use_depth
                                    + (1.0 - self._llm_depth_ema_alpha) * self._last_llm_depth
                                )
                            self.env.unwrapped.llm_depth = self._last_llm_depth
                step_in_ep += 1

                action = self.select_action(state, mask_t)
                act_int = action.item()
                ep_actions.append(act_int)

                obs, reward, terminated, truncated, info = self.env.step(act_int)
                nxt_mask = info.get("mask", None)


                new_depth = float(self.env.unwrapped.depth_history[-1]) \
                    if getattr(self.env.unwrapped, "depth_history", None) else None
                if new_depth is not None:
                    eb = _explore_bin_bonus(new_depth, visited_bins)
                    if eb > 0.0:
                        reward += eb
                        ep_explore_bonus += eb

                ep_ret  += reward
                r_t      = torch.tensor([reward], device=self.device)

                obs = np.nan_to_num(obs.flatten())
                if terminated:
                    next_state = None
                    nxt_mask_t = None
                else:
                    next_state = torch.tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    nxt_mask_t = (
                        torch.tensor(nxt_mask, dtype=torch.bool, device=self.device)
                        if nxt_mask is not None else None
                    )

                cur_mask_t = (
                    torch.tensor(cur_mask, dtype=torch.bool, device=self.device)
                    if cur_mask is not None else None
                )

                leg_buf.append((state, action, next_state, r_t, cur_mask_t, nxt_mask_t))
                leg_depths.append(cur_depth)


                is_turn = (prev_action != -1 and act_int != prev_action)
                ep_ended = terminated or truncated
                if (is_turn or ep_ended) and leg_buf:
                    bonus = 0.0


                    if len(leg_depths) >= LLM_MIN_LEG_LEN:
                        leg_span = float(max(leg_depths) - min(leg_depths))
                        span_bonus = LEG_SPAN_BONUS_GAIN * min(leg_span, LEG_SPAN_CAP)
                        bonus += span_bonus
                        ep_explore_bonus += span_bonus


                    if (not math.isnan(self._last_llm_depth)
                            and len(leg_depths) >= LLM_MIN_LEG_LEN):

                        leg_mid = float(np.mean(leg_depths))
                        mid_dist = abs(leg_mid - self._last_llm_depth)


                        if self.llm_bonus > 0:
                            lo = self._last_llm_depth - self.llm_bonus_half_width
                            hi = self._last_llm_depth + self.llm_bonus_half_width
                            if min(leg_depths) <= hi and max(leg_depths) >= lo:
                                bonus += self.llm_bonus
                                ep_llm_bonus_count += 1


                        if mid_dist < LLM_MID_DIST_MAX:
                            bonus += LLM_MID_BONUS * (LLM_MID_DIST_MAX - mid_dist) / LLM_MID_DIST_MAX


                    for i, (s, a, ns, r, cm, nm) in enumerate(leg_buf):
                        is_last = (i == len(leg_buf) - 1)
                        r_push = r + bonus if (bonus > 0 and is_last) else r
                        if ns is not None:
                            self.replay_memory.push(s, a, ns, r_push, cm, nm)

                    leg_buf   = []
                    leg_depths = []

                prev_action = act_int
                state    = next_state
                cur_mask = nxt_mask

                res = self.optimize_model()
                if res is not None:
                    ep_loss  += res[0]
                    ep_td    += res[1]
                    ep_kl_rl += res[2]

                self._soft_update()

                if ep_ended:
                    ep_len = max(t + 1, 1)
                    self.recent_losses.append(ep_loss / ep_len)
                    self.recent_returns.append(ep_ret)
                    self.recent_lengths.append(ep_len)

                    turn_depths = [
                        ep_depths[i]
                        for i in range(1, len(ep_actions))
                        if ep_actions[i] != ep_actions[i - 1]
                    ]
                    avg_turn_depth = np.mean(turn_depths) if turn_depths else float('nan')


                    max_grad_depth = self.env.unwrapped.get_max_grad_depth()
                    max_grad_depth = max_grad_depth if max_grad_depth is not None else float('nan')


                    depth_gap = (
                        abs(avg_turn_depth - self._last_llm_depth)
                        if not math.isnan(avg_turn_depth)
                           and not math.isnan(self._last_llm_depth)
                        else float('nan')
                    )



                    llm_bias = (
                        self._last_llm_depth - max_grad_depth
                        if not math.isnan(max_grad_depth)
                           and not math.isnan(self._last_llm_depth)
                        else float('nan')
                    )
                    if not math.isnan(llm_bias):
                        self.recent_llm_bias.append(llm_bias)

                    self._i_ep += 1
                    w_rl_cur = self._kl_weight

                    eps = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                          math.exp(-1.0 * self.steps_done / self.EPS_DECAY)
                    kl_rl_actual = ep_kl_rl / ep_len
                    self.writer.add_scalar("Training/Epsilon",      eps,             i_ep)
                    self.writer.add_scalar("Training/TDLoss",       ep_td / ep_len,  i_ep)
                    self.writer.add_scalar("Training/KLRLLoss",     kl_rl_actual,    i_ep)
                    self.writer.add_scalar("Training/LLMBonusLegs", ep_llm_bonus_count, i_ep)
                    self.writer.add_scalar("Training/ExploreBonus", ep_explore_bonus, i_ep)
                    self.writer.add_scalar("Training/DepthGap",     depth_gap,       i_ep)
                    swanlab.log({
                        "ep/epsilon":         eps,
                        "ep/td_loss":         ep_td  / ep_len,
                        "ep/kl_rl":           kl_rl_actual,
                        "ep/llm_depth":       self._last_llm_depth,
                        "ep/turn_depth":      avg_turn_depth,
                        "ep/max_grad_depth":  max_grad_depth,
                        "ep/depth_gap":       depth_gap,
                        "ep/llm_bias":        llm_bias,
                        "ep/w_rl":            w_rl_cur,
                        "ep/llm_bonus_legs":  ep_llm_bonus_count,
                        "ep/explore_bonus":   ep_explore_bonus,
                    }, step=i_ep)

                    if (i_ep + 1) % log_interval == 0:
                        al = np.mean(self.recent_losses[-log_interval:])
                        ar = np.mean(self.recent_returns[-log_interval:])
                        gap_str  = f"{depth_gap:.1f}m" if not math.isnan(depth_gap) else "nan"
                        mgd_str  = f"{max_grad_depth:.1f}m" if not math.isnan(max_grad_depth) else "nan"
                        avg_bias = np.mean(self.recent_llm_bias[-log_interval:]) \
                                   if self.recent_llm_bias else float('nan')
                        bias_str = f"{avg_bias:+.1f}m" if not math.isnan(avg_bias) else "nan"
                        print(
                            f"Ep {i_ep+1:5d}  "
                            f"loss={al:.4f}  "
                            f"kl_rl={kl_rl_actual:.4f}(w={w_rl_cur:.1f})  "
                            f"return={ar:.2f}  "
                            f"llm_z={self._last_llm_depth:.1f}m  "
                            f"grad_z={mgd_str}  "
                            f"llm_bias={bias_str}  "
                            f"gap={gap_str}  "
                            f"bonus_legs={ep_llm_bonus_count}  "
                            f"explore={ep_explore_bonus:.0f}  "
                            f"t={time.time()-t0:.0f}s"
                        )
                        self.writer.add_scalar("Training/Avg_Loss",   al, i_ep)
                        self.writer.add_scalar("Training/Avg_Return", ar, i_ep)
                        swanlab.log({
                            f"avg{log_interval}/loss":   al,
                            f"avg{log_interval}/return": ar,
                        }, step=i_ep)
                    break

        print("训练完成。")
        self.writer.close()






if __name__ == "__main__":

    _PRETRAIN_PATH = os.path.join(LOG_DIR_MODELSAVE, "myDQNmodelAttn202603121756.pth")

    log_dir = LOG_DIR_TENSORBOARD + "LLMRewardShape" + time.strftime("%Y%m%d%H%M")
    env     = gym.make(ENV_ID, render_mode=RENDER_MODE)

    swanlab.init(
        project        = SWANLAB_PROJECT_LLM_DQN,
        experiment_name= SWANLAB_EXPERIMENT_LLM_DQN + "-" + time.strftime("%m%d%H%M"),
        config={

            "lr":                    DQN_FT_LR,
            "gamma":                 DQN_GAMMA,
            "tau":                   DQN_TAU,
            "eps_start":             DQN_FT_EPS_START,
            "eps_end":               DQN_EPS_END,
            "eps_decay":             DQN_EPS_DECAY,
            "batch_size":            DQN_BATCH_SIZE,
            "seq_len":               SEQ_LEN,
            "hidden_size":           DQN_HIDDEN_SIZE,
            "num_heads":             NUM_HEADS,
            "num_layers":            NUM_LAYERS,
            "replay_memory_size":    REPLAY_MEMORY_SIZE,
            "max_episode_len":       500,

            "kl_lambda_rl_start":    KL_LAMBDA_RL_START,
            "kl_lambda_rl_floor":    KL_LAMBDA_RL_FLOOR,
            "kl_anneal_eps":         KL_ANNEAL_EPS,
            "kl_policy_tau":         KL_POLICY_TAU,

            "llm_bonus":             LLM_BONUS,
            "llm_bonus_half_width":  LLM_BONUS_HALF_WIDTH,
            "llm_mid_bonus":         LLM_MID_BONUS,
            "llm_mid_dist_max":      LLM_MID_DIST_MAX,
            "llm_infer_interval":    LLM_INFER_INTERVAL,
            "llm_min_leg_len":       LLM_MIN_LEG_LEN,

            "max_episode_len":       MAX_EPISODES_LEN,
            "training_episodes":     TRAINING_EPISODES,
            "manual_seed":           MANUAL_SEED,
            "cuda_device":           CUDA_DEVICE_DQN,
            "pretrain_path":         _PRETRAIN_PATH,
        },
    )

    agent = LLMRewardShapeDQNAgent(
        env,
        use_cuda=True,
        log_dir=log_dir,
        model_path             = _PRETRAIN_PATH,
        kl_policy_tau          = KL_POLICY_TAU,
        kl_lambda_rl_start     = KL_LAMBDA_RL_START,
        kl_lambda_rl_floor     = KL_LAMBDA_RL_FLOOR,
        kl_anneal_eps          = KL_ANNEAL_EPS,
        llm_bonus              = LLM_BONUS,
        llm_bonus_half_width   = LLM_BONUS_HALF_WIDTH,
        llm_infer_interval     = LLM_INFER_INTERVAL,
        finetune_lr            = DQN_FT_LR,
        finetune_eps_start     = DQN_FT_EPS_START,


    )
    check_data_overlap()

    num_episodes = (
        TRAINING_EPISODES
        if torch.cuda.is_available() or torch.backends.mps.is_available()
        else 50
    )
    agent.train(num_episodes, log_interval=LOG_INTERVAL)

    save_path = LOG_DIR_MODELSAVE + "LLMRewardShape" + time.strftime("%Y%m%d%H%M") + ".pth"
    agent.save(save_path)
    swanlab.finish()

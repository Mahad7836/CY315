"""
train.py
Trains 4 SAC agents, one per Eve CSI noise level:
  1. noise = 0.0m  — perfect Eve CSI (paper baseline)
  2. noise = 2.0m  — slight location uncertainty
  3. noise = 5.0m  — moderate location uncertainty
  4. noise = 10.0m — severe location uncertainty (our contribution)

Models saved to models/sac_noise_<sigma>.zip
Results (reward curves) saved to results/
"""

import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.cfj_env import WirelessJammingEnv
from uasac import UASAC

# ── Max CPU performance ────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"]  = "8"
os.environ["MKL_NUM_THREADS"]  = "8"
os.environ["OPENBLAS_NTHREADS"] = "8"
torch.set_num_threads(8)

# Raise Windows process priority to High
try:
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00000080  # HIGH_PRIORITY_CLASS
    )
    print("Process priority: HIGH")
except Exception:
    pass

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

TIMESTEPS   = 50_000
NOISE_LEVELS = [0.0, 2.0, 5.0, 10.0]   # metres of Eve location uncertainty

# ── Callback to track reward per episode ──────────────────────────────
class RewardLogger(BaseCallback):
    """Tracks per-episode reward across ALL parallel envs, not just env 0."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = None

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        if self._ep_reward is None:
            self._ep_reward = np.zeros(len(rewards))
        self._ep_reward += rewards
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self._ep_reward[i])
                self._ep_reward[i] = 0.0
        return True


def train_agent(noise_std: float, timesteps: int = TIMESTEPS):
    """Train one SAC agent for a given Eve CSI noise level."""
    env      = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                   csi_noise_std=noise_std)
    callback = RewardLogger()

    model = SAC(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),  # 9-layer equiv hidden units
    )
    model.learn(total_timesteps=timesteps, callback=callback)

    tag = f"noise_{noise_std:.1f}"
    model.save(f"models/sac_{tag}")
    print(f"  [noise={noise_std}m] trained and saved.")

    return model, callback.episode_rewards


# ── Phase 1: Train for each noise level (LEGACY — models already saved) ──
# These 4 agents are superseded by UA-SAC. Only uncomment if you need to
# regenerate sac_noise_*.zip from scratch.
#
# all_rewards = {}
# for noise in NOISE_LEVELS:
#     label = "Baseline (perfect CSI)" if noise == 0.0 else f"Imperfect CSI σ={noise}m"
#     print(f"\n→ {label}")
#     _, rewards = train_agent(noise_std=noise)
#     all_rewards[noise] = rewards
#
# plt.figure(figsize=(8, 4.5))
# for noise, rewards in all_rewards.items():
#     label = "Baseline (perfect CSI)" if noise == 0.0 else f"σ_ε = {noise} m"
#     window = max(1, len(rewards) // 50)
#     smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
#     plt.plot(smoothed, label=label)
# plt.xlabel("Episode", fontsize=12)
# plt.ylabel("Sum Secrecy Capacity (bps/Hz)", fontsize=12)
# plt.title("Training Convergence — Baseline vs. Imperfect Eve CSI", fontsize=11)
# plt.legend(fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("results/training_convergence.png", dpi=150)
# print("\nSaved: results/training_convergence.png")


# ══════════════════════════════════════════════════════════════════════
# Phase 2 — UA-SAC (Uncertainty-Aware SAC)
#
# Single universal agent trained across all σ levels simultaneously.
# State: ℝ¹⁵  (adds ρ = σ/D_max as element 15)
# Reward: worst-case over M=5 sampled Eve locations per step
# Run separately — takes ~6-8 hrs on CPU (100k timesteps, M=5 reward)
#
# To train: uncomment the call at the bottom of this block and run:
#   venv/Scripts/python train.py
# ══════════════════════════════════════════════════════════════════════

def make_env():
    def _init():
        return WirelessJammingEnv(
            num_aps=4, num_users=2, num_eves=1,
            sigma_range=(0.0, 10.0),
            M=5,
            beta=1.0,
            augment_rho=True,
        )
    return _init


def train_uasac(timesteps: int = 100_000):
    N_ENVS = 4   # parallel env workers — leaves 4 cores for PyTorch training
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = UASAC(
        "MlpPolicy", env,
        beta=1.0,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,      # larger buffer to match faster data collection
        batch_size=256,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callback = RewardLogger()
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save("models/uasac_robust")
    model.save_ent_history("results/uasac_ent_history.npz")
    np.save("results/uasac_reward_history.npy", np.array(callback.episode_rewards))
    print("UA-SAC saved → models/uasac_robust")

    window   = max(1, len(callback.episode_rewards) // 50)
    smoothed = np.convolve(callback.episode_rewards,
                           np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4.5))
    plt.plot(smoothed, color="#8b5cf6", linewidth=2, label="UA-SAC (σ ~ U[0,10])")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Worst-Case Sum Secrecy (bps/Hz)", fontsize=12)
    plt.title("UA-SAC Training Convergence", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/uasac_convergence.png", dpi=150)
    print("Saved: results/uasac_convergence.png")


if __name__ == "__main__":
    train_uasac(timesteps=100_000)
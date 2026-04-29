"""
train.py
========

Train SAC agents at multiple Eve-CSI noise levels.

For each noise level sigma in NOISE_LEVELS:
    - build a WirelessJammingEnv with csi_noise_std=sigma
    - train a Soft Actor-Critic policy for TIMESTEPS environment steps
    - save the model to models/sac_noise_<sigma>.zip
    - log the per-episode return

A combined training-curve plot is written to results/training_convergence.png.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from env.cfj_env import WirelessJammingEnv


# ----------------------------- Defaults -----------------------------
TIMESTEPS     = 50_000
NOISE_LEVELS  = [0.0, 2.0, 5.0, 10.0]   # metres of Eve-location uncertainty
NUM_APS       = 4
NUM_USERS     = 2
NUM_EVES      = 1
SEED          = 42

MODEL_DIR     = "models"
RESULTS_DIR   = "results"


def set_global_seed(seed: int) -> None:
    """Seed numpy, random, and torch (if available) for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RewardLogger(BaseCallback):
    """Track per-episode returns for plotting."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 wraps single envs in a length-1 list internally
        self._ep_reward += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True


def train_one(noise_std: float,
              timesteps: int,
              num_aps: int,
              num_users: int,
              num_eves: int,
              seed: int) -> tuple[SAC, list[float]]:
    """Train a single SAC agent at the given Eve-CSI noise level."""
    env = WirelessJammingEnv(
        num_aps=num_aps,
        num_users=num_users,
        num_eves=num_eves,
        csi_noise_std=noise_std,
        seed=seed,
    )

    callback = RewardLogger()
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=timesteps, callback=callback)

    tag = f"noise_{noise_std:.1f}"
    save_path = os.path.join(MODEL_DIR, f"sac_{tag}")
    model.save(save_path)
    print(f"  [sigma={noise_std:>4.1f} m] saved -> {save_path}.zip "
          f"({len(callback.episode_rewards)} episodes, "
          f"final-100-mean={np.mean(callback.episode_rewards[-100:]):.2f})")
    return model, callback.episode_rewards


def plot_convergence(rewards_by_noise: dict[float, list[float]],
                     out_path: str) -> None:
    plt.figure(figsize=(8, 4.5))
    for noise, rewards in rewards_by_noise.items():
        if not rewards:
            continue
        label = ("Baseline (perfect CSI)" if noise == 0.0
                 else f"sigma_eps = {noise:g} m")
        window = max(1, len(rewards) // 50)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(smoothed, label=label, linewidth=1.6)
    plt.xlabel("Episode")
    plt.ylabel("Sum Secrecy Capacity (bps/Hz)")
    plt.title("Training Convergence — SAC under varying Eve-CSI noise")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS,
                        help=f"Training steps per noise level (default {TIMESTEPS})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-aps", type=int, default=NUM_APS)
    parser.add_argument("--num-users", type=int, default=NUM_USERS)
    parser.add_argument("--num-eves", type=int, default=NUM_EVES)
    parser.add_argument("--noise", type=float, nargs="+", default=NOISE_LEVELS,
                        help="One or more Eve CSI noise std values to train at")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    set_global_seed(args.seed)

    print("=" * 60)
    print(f"Training SAC | timesteps={args.timesteps} | seed={args.seed} | "
          f"APs={args.num_aps} | users={args.num_users} | eves={args.num_eves}")
    print("=" * 60)

    rewards_by_noise: dict[float, list[float]] = {}
    for noise in args.noise:
        label = ("Baseline (perfect CSI)" if noise == 0.0
                 else f"Imperfect CSI sigma={noise:g} m")
        print(f"\n--> {label}")
        _, ep_rewards = train_one(
            noise_std=float(noise),
            timesteps=args.timesteps,
            num_aps=args.num_aps,
            num_users=args.num_users,
            num_eves=args.num_eves,
            seed=args.seed,
        )
        rewards_by_noise[float(noise)] = ep_rewards

    plot_convergence(rewards_by_noise,
                     os.path.join(RESULTS_DIR, "training_convergence.png"))


if __name__ == "__main__":
    main()

"""
generate_plots_obs_aware.py
===========================

Better proxy plots than generate_plots.py for the *novelty* figures.

The earlier proxy (random search) doesn't condition on observations, so
its noisy-CSI plots are flat — random search can't be fooled by noisy
Eve coordinates because it ignores them. That's a true upper bound on
performance but a bad demo of why imperfect CSI matters.

This script trains a tiny observation-aware policy via Cross-Entropy
Method (CEM): a small linear map from observation -> action, optimised
per noise level by sampling weight matrices and keeping the top
performers. CEM is gradient-free so it runs without torch.

Output files match test.py's filenames so they can be used directly in
the writeup as placeholders until real SAC training runs.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from env.cfj_env import WirelessJammingEnv, MAX_POWER_W

RESULTS_DIR = "results_obs_aware"
N_EPISODES_TRAIN = 30          # rollouts per CEM iteration
N_EPISODES_EVAL  = 100
CEM_ITERS        = 20
CEM_POP          = 60
CEM_ELITE        = 12
SEED             = 42


def linear_policy(theta: np.ndarray, obs: np.ndarray, num_aps: int) -> np.ndarray:
    """theta has shape (obs_dim, num_aps); action = tanh(obs @ theta)."""
    return np.tanh(obs @ theta).astype(np.float32)


def evaluate_theta(theta: np.ndarray,
                   env: WirelessJammingEnv,
                   n_episodes: int,
                   start_seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean sum_secrecy, mean sum_eve, mean secrecy_ratio)."""
    secs, eves, ratios = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=start_seed + ep)
        action = linear_policy(theta, obs, env.num_aps)
        m = env.evaluate_policy(action)
        secs.append(m["sum_secrecy_capacity"])
        eves.append(m["sum_eve_capacity"])
        ratios.append(m["secrecy_ratio"])
    return float(np.mean(secs)), float(np.mean(eves)), float(np.mean(ratios)) * 100.0


def cem_train(env: WirelessJammingEnv,
              iters: int = CEM_ITERS,
              pop: int = CEM_POP,
              elite: int = CEM_ELITE,
              n_eps: int = N_EPISODES_TRAIN,
              seed: int = SEED) -> np.ndarray:
    """Cross-Entropy Method for the linear policy."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.num_aps
    rng = np.random.default_rng(seed)
    mean = np.zeros((obs_dim, act_dim), dtype=np.float32)
    std  = np.ones ((obs_dim, act_dim), dtype=np.float32) * 0.5

    best_score = -np.inf
    best_theta = mean.copy()
    for it in range(iters):
        # Sample population
        thetas = rng.normal(loc=mean, scale=std, size=(pop, obs_dim, act_dim)).astype(np.float32)
        scores = np.array([evaluate_theta(t, env, n_eps, start_seed=it * 1000)[0]
                           for t in thetas])
        # Update on top elite
        top = np.argsort(scores)[-elite:]
        elite_set = thetas[top]
        mean = elite_set.mean(axis=0)
        std  = elite_set.std (axis=0) + 1e-3   # floor to keep exploration alive
        if scores[top[-1]] > best_score:
            best_score = float(scores[top[-1]])
            best_theta = thetas[top[-1]].copy()
    return best_theta


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("Training observation-aware policies via CEM "
          f"(iters={CEM_ITERS}, pop={CEM_POP}, elite={CEM_ELITE})\n")

    # Train one policy per noise level
    NOISE_TRAIN = [0.0, 2.0, 5.0, 10.0]
    thetas: dict[float, np.ndarray] = {}
    for sigma in NOISE_TRAIN:
        env = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                 csi_noise_std=sigma, seed=SEED)
        print(f"  CEM-training at sigma={sigma}m ...", flush=True)
        theta = cem_train(env, seed=SEED + int(sigma * 100))
        sec, eve, ratio = evaluate_theta(theta, env, N_EPISODES_EVAL, start_seed=99_999)
        thetas[sigma] = theta
        print(f"    eval at sigma={sigma}m  sec={sec:.2f}  eve={eve:.2f}  "
              f"ratio={ratio:.0f}%")

    # ---- Plot 2 & 3: secrecy & ratio vs noise (uses nearest-trained-sigma) ----
    print("\nEvaluating on noise sweep ...")
    NOISE_EVAL = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    sec_y, eve_y, ratio_y = [], [], []
    for sigma in NOISE_EVAL:
        env = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                 csi_noise_std=sigma, seed=SEED)
        nearest = min(NOISE_TRAIN, key=lambda x: abs(x - sigma))
        theta = thetas[nearest]
        sec, eve, ratio = evaluate_theta(theta, env, N_EPISODES_EVAL, start_seed=42_424)
        sec_y.append(sec); eve_y.append(eve); ratio_y.append(ratio)
        print(f"  sigma_eval={sigma:>4.1f}m  (theta_sigma={nearest:>4.1f}m)  "
              f"sec={sec:.2f}  ratio={ratio:.0f}%  eve={eve:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(NOISE_EVAL, sec_y, "g-^", ms=7, linewidth=2, label="Sum secrecy (CEM policy)")
    ax.plot(NOISE_EVAL, eve_y, "r--o", ms=6, linewidth=1.5, label="Sum Eve capacity")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7, label="Perfect CSI")
    ax.set_xlabel(r"Eve location uncertainty $\sigma_\epsilon$ (m)")
    ax.set_ylabel("Capacity (bps/Hz)")
    ax.set_title("Effect of Imperfect Eve CSI — CEM observation-aware policy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot2_secrecy_vs_noise.png", dpi=150)
    plt.close()
    print(f"  saved -> {RESULTS_DIR}/plot2_secrecy_vs_noise.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(NOISE_EVAL, ratio_y, "m-D", ms=7, linewidth=2)
    ax.set_xlabel(r"Eve location uncertainty $\sigma_\epsilon$ (m)")
    ax.set_ylabel("Secrecy Ratio (%)")
    ax.set_title("Secrecy Ratio vs. Imperfect Eve CSI — CEM policy")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot3_ratio_vs_noise.png", dpi=150)
    plt.close()
    print(f"  saved -> {RESULTS_DIR}/plot3_ratio_vs_noise.png")

    # ---- Print final summary the user can copy into the report ----
    print("\nCEM-policy summary table (4 APs, 2 users, 1 Eve, "
          f"{N_EPISODES_EVAL} episodes per cell):")
    print(f"{'sigma_eval (m)':>14s} {'SecCap':>8s} {'EveCap':>8s} {'Ratio':>8s}")
    for s, sec, eve, ratio in zip(NOISE_EVAL, sec_y, eve_y, ratio_y):
        print(f"{s:>14.1f} {sec:>8.2f} {eve:>8.2f} {ratio:>7.0f}%")


if __name__ == "__main__":
    main()

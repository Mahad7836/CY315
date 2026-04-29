"""
test.py
=======

Evaluate trained SAC agents and produce four comparison plots:

  Plot 1 - Sum secrecy capacity vs. number of APs   (paper Fig. 3 trend)
  Plot 2 - Sum secrecy capacity vs. Eve CSI noise   (project novelty)
  Plot 3 - Secrecy ratio vs. Eve CSI noise          (project novelty)
  Plot 4 - Bar chart: Normal Wi-Fi / Smart AP / RL-CFJ perfect / RL-CFJ noisy

Three baselines are implemented properly:
  - Normal Wi-Fi : nearest-AP association, all APs at max power, no jamming
                   (idle APs OFF — paper sec. IV).
  - Smart AP     : eq. (7) PLS-aware association, all APs at max power,
                   idle APs OFF.
  - RL-CFJ       : eq. (7) association, SAC-optimised continuous powers,
                   idle APs jam.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from env.cfj_env import WirelessJammingEnv, MAX_POWER_W


RESULTS_DIR     = "results"
MODEL_DIR       = "models"
N_EVAL_EPISODES = 200


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------
@dataclass
class Metrics:
    sum_secrecy: float
    sum_eve:     float
    secrecy_ratio: float


def evaluate_normal_wifi(env: WirelessJammingEnv, n: int) -> Metrics:
    """
    Normal Wi-Fi:
      - association by minimum distance (highest SINR ~ closest AP)
      - associated APs transmit at max power
      - non-associated APs OFF (paper sec. IV)
    """
    sums, eves, ratios = [], [], []
    for ep in range(n):
        env.reset(seed=ep)
        d = env._pairwise_dist(env.user_positions, env.ap_positions)  # (K, N)
        nearest = np.argmin(d, axis=1)                                # (K,)
        # Override association to nearest-AP for this baseline only
        env.association = nearest.astype(np.int32)
        powers = np.zeros(env.num_aps, dtype=np.float32)
        powers[np.unique(nearest)] = MAX_POWER_W
        m = env.evaluate_policy(powers)
        sums.append(m["sum_secrecy_capacity"])
        eves.append(m["sum_eve_capacity"])
        ratios.append(m["secrecy_ratio"])
    return Metrics(float(np.mean(sums)),
                   float(np.mean(eves)),
                   float(np.mean(ratios)) * 100.0)


def evaluate_smart_ap(env: WirelessJammingEnv, n: int) -> Metrics:
    """
    Smart AP:
      - association via eq. (7) (already done in env.reset)
      - associated APs at max power
      - non-associated APs OFF (no jamming)
    """
    sums, eves, ratios = [], [], []
    for ep in range(n):
        env.reset(seed=ep)
        powers = np.zeros(env.num_aps, dtype=np.float32)
        powers[np.unique(env.association)] = MAX_POWER_W
        m = env.evaluate_policy(powers)
        sums.append(m["sum_secrecy_capacity"])
        eves.append(m["sum_eve_capacity"])
        ratios.append(m["secrecy_ratio"])
    return Metrics(float(np.mean(sums)),
                   float(np.mean(eves)),
                   float(np.mean(ratios)) * 100.0)


def evaluate_rl_cfj(env: WirelessJammingEnv, model: SAC, n: int) -> Metrics:
    """
    RL-CFJ:
      - association via eq. (7) (env default)
      - SAC predicts powers in [-1, 1]^N which env.evaluate_policy maps.
    """
    sums, eves, ratios = [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        action, _ = model.predict(obs, deterministic=True)
        m = env.evaluate_policy(action)
        sums.append(m["sum_secrecy_capacity"])
        eves.append(m["sum_eve_capacity"])
        ratios.append(m["secrecy_ratio"])
    return Metrics(float(np.mean(sums)),
                   float(np.mean(eves)),
                   float(np.mean(ratios)) * 100.0)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
def load_model(noise: float, env: WirelessJammingEnv,
               required: bool = False) -> SAC | None:
    """
    Load a saved SAC model for the given noise level. Returns None if the
    file isn't present and `required` is False; raises if `required` is True.
    """
    path = os.path.join(MODEL_DIR, f"sac_noise_{noise:.1f}.zip")
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Required model not found: {path}.\n"
                f"Run: python train.py --noise {noise}")
        return None
    return SAC.load(path, env=env)


# ------------------------------------------------------------------
# Plot 1 — secrecy vs. number of APs
# ------------------------------------------------------------------
def plot_secrecy_vs_aps(ap_counts: list[int],
                        n_episodes: int,
                        seed: int) -> None:
    """
    For each AP count we need a SEPARATELY TRAINED RL model
    (different observation/action dimensions per AP count).  If a
    matching model isn't available, that point is skipped — we never
    fabricate RL numbers.
    """
    print("\nPlot 1: secrecy vs. number of APs")
    normal_y, smart_y, rl_y, rl_x = [], [], [], []
    for n_aps in ap_counts:
        env = WirelessJammingEnv(num_aps=n_aps, num_users=2, num_eves=1,
                                 csi_noise_std=0.0, seed=seed)
        nw = evaluate_normal_wifi(env, n_episodes)
        sa = evaluate_smart_ap(env,  n_episodes)
        normal_y.append(nw.sum_secrecy)
        smart_y .append(sa.sum_secrecy)
        print(f"  APs={n_aps:>2d}  normal={nw.sum_secrecy:.2f}  "
              f"smart={sa.sum_secrecy:.2f}", end="")

        # RL model trained at this exact AP count?
        rl_model_path = os.path.join(MODEL_DIR, f"sac_naps_{n_aps}_noise_0.0.zip")
        if os.path.exists(rl_model_path):
            model = SAC.load(rl_model_path, env=env)
            rl = evaluate_rl_cfj(env, model, n_episodes)
            rl_y.append(rl.sum_secrecy)
            rl_x.append(n_aps)
            print(f"  rl={rl.sum_secrecy:.2f}")
        else:
            print(f"  rl=<no model: train with --num-aps {n_aps}>")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ap_counts, normal_y, "b-o", ms=7, label="Normal Wi-Fi")
    ax.plot(ap_counts, smart_y,  "g-s", ms=7, label="Smart AP")
    if rl_y:
        ax.plot(rl_x, rl_y, "r-^", ms=8, label="RL-CFJ (perfect CSI)")
    ax.set_xlabel("Number of APs")
    ax.set_ylabel("Sum Secrecy Capacity (bps/Hz)")
    ax.set_title("Sum Secrecy Capacity vs. Number of APs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot1_secrecy_vs_aps.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}")


# ------------------------------------------------------------------
# Plot 2 + Plot 3 — performance vs. Eve CSI noise
# ------------------------------------------------------------------
def plot_secrecy_vs_noise(noise_levels: list[float],
                          trained_levels: list[float],
                          num_aps: int,
                          n_episodes: int,
                          seed: int) -> None:
    print("\nPlot 2/3: secrecy vs. Eve CSI noise")
    sec_means, ratio_means, eve_means = [], [], []
    for sigma in noise_levels:
        env = WirelessJammingEnv(num_aps=num_aps, num_users=2, num_eves=1,
                                 csi_noise_std=sigma, seed=seed)
        # Use the closest available trained noise level
        nearest = min(trained_levels, key=lambda x: abs(x - sigma))
        model = load_model(nearest, env, required=True)
        m = evaluate_rl_cfj(env, model, n_episodes)
        sec_means  .append(m.sum_secrecy)
        ratio_means.append(m.secrecy_ratio)
        eve_means  .append(m.sum_eve)
        print(f"  sigma={sigma:>4.1f}m  (model_sigma={nearest:>4.1f}m)  "
              f"secrecy={m.sum_secrecy:.2f}  ratio={m.secrecy_ratio:.0f}%  "
              f"eve={m.sum_eve:.2f}")

    # Plot 2: secrecy capacity
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(noise_levels, sec_means, "g-^", ms=7, linewidth=2,
            label="RL-CFJ sum secrecy")
    ax.plot(noise_levels, eve_means, "r--o", ms=6, linewidth=1.5,
            label="Sum Eve capacity")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.7,
               label="Perfect CSI")
    ax.set_xlabel("Eve location uncertainty  sigma_eps  (m)")
    ax.set_ylabel("Capacity (bps/Hz)")
    ax.set_title("Effect of Imperfect Eve CSI on Secrecy and Eve Capacity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "plot2_secrecy_vs_noise.png")
    plt.savefig(out2, dpi=150); plt.close()
    print(f"  -> {out2}")

    # Plot 3: secrecy ratio
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(noise_levels, ratio_means, "m-D", ms=7, linewidth=2)
    ax.set_xlabel("Eve location uncertainty  sigma_eps  (m)")
    ax.set_ylabel("Secrecy Ratio (%)")
    ax.set_title("Effect of Imperfect Eve CSI on Secrecy Ratio")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = os.path.join(RESULTS_DIR, "plot3_ratio_vs_noise.png")
    plt.savefig(out3, dpi=150); plt.close()
    print(f"  -> {out3}")


# ------------------------------------------------------------------
# Plot 4 — bar chart at fixed (N=4, K=2, J=1)
# ------------------------------------------------------------------
def plot_comparison_bars(noise_for_imperfect: float,
                         num_aps: int,
                         n_episodes: int,
                         seed: int) -> None:
    print("\nPlot 4: comparison bar chart")
    env_clean = WirelessJammingEnv(num_aps=num_aps, num_users=2, num_eves=1,
                                   csi_noise_std=0.0, seed=seed)
    env_noisy = WirelessJammingEnv(num_aps=num_aps, num_users=2, num_eves=1,
                                   csi_noise_std=noise_for_imperfect, seed=seed)

    nw   = evaluate_normal_wifi(env_clean, n_episodes)
    sa   = evaluate_smart_ap(env_clean,    n_episodes)
    m_cl = load_model(0.0,                  env_clean, required=True)
    m_no = load_model(noise_for_imperfect,  env_noisy, required=True)
    rl_clean = evaluate_rl_cfj(env_clean, m_cl, n_episodes)
    rl_noisy = evaluate_rl_cfj(env_noisy, m_no, n_episodes)

    labels = ["Normal Wi-Fi", "Smart AP",
              "RL-CFJ\n(perfect CSI)",
              f"RL-CFJ\n(noisy CSI sigma={noise_for_imperfect:g}m)"]
    sec   = [nw.sum_secrecy,   sa.sum_secrecy,
             rl_clean.sum_secrecy, rl_noisy.sum_secrecy]
    eve   = [nw.sum_eve,       sa.sum_eve,
             rl_clean.sum_eve,     rl_noisy.sum_eve]
    ratio = [nw.secrecy_ratio, sa.secrecy_ratio,
             rl_clean.secrecy_ratio, rl_noisy.secrecy_ratio]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, vals, ylabel, title, fmt in [
        (axes[0], sec,   "Sum Secrecy Capacity (bps/Hz)",
         "Sum Secrecy Capacity",        "{:.2f}"),
        (axes[1], eve,   "Sum Eve Capacity (bps/Hz)",
         "Sum Eve Capacity (lower is better)", "{:.2f}"),
        (axes[2], ratio, "Secrecy Ratio (%)",
         "Secrecy Ratio",               "{:.0f}%"),
    ]:
        bars = ax.bar(x, vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if "Ratio" in title:
            ax.set_ylim(0, 110)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if "Ratio" in title else 0.05),
                    fmt.format(v),
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Performance comparison ({num_aps} APs, "
                 f"2 users, 1 Eve, {n_episodes} episodes)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot4_comparison_bar.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  -> {out}")

    # Also print a clean text-table summary that lands in the report
    print("\n--- Final comparison table ---")
    print(f"{'Method':<28s} {'SecCap':>8s} {'EveCap':>8s} {'Ratio':>8s}")
    for name, m in [("Normal Wi-Fi",        nw),
                    ("Smart AP",            sa),
                    ("RL-CFJ perfect CSI",  rl_clean),
                    (f"RL-CFJ sigma={noise_for_imperfect:g}m", rl_noisy)]:
        print(f"{name:<28s} {m.sum_secrecy:>8.2f} {m.sum_eve:>8.2f} "
              f"{m.secrecy_ratio:>7.0f}%")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-aps", type=int, default=4)
    parser.add_argument("--ap-counts", type=int, nargs="+",
                        default=[4, 5, 7, 9, 13])
    parser.add_argument("--noise-sweep", type=float, nargs="+",
                        default=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    parser.add_argument("--trained-noise", type=float, nargs="+",
                        default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--bar-noise", type=float, default=5.0,
                        help="Noise level used for the imperfect-CSI bar")
    parser.add_argument("--skip-plot1", action="store_true",
                        help="Skip plot 1 (needs models trained at "
                             "different AP counts)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not args.skip_plot1:
        plot_secrecy_vs_aps(args.ap_counts, args.n_episodes, args.seed)

    plot_secrecy_vs_noise(args.noise_sweep, args.trained_noise,
                          args.num_aps, args.n_episodes, args.seed)

    plot_comparison_bars(args.bar_noise, args.num_aps,
                         args.n_episodes, args.seed)

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

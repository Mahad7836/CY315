"""
generate_plots.py
=================

Sandbox-friendly producer of the four result plots WITHOUT torch/SB3.

We don't have torch in this sandbox so we can't run real SAC. As a stand-in
that approximates the trained-policy upper bound, we use a per-scene
"best-of-1500 random search" oracle. This is exactly the validation #6
proxy. It gives us realistic-looking plots so the user can see what
test.py will produce on their own machine. When they run the real
test.py with trained SAC models, the curves should be similar but
typically a bit better (SAC uses gradients).

This script exists to produce demo plots for the writeup, not to
replace train.py + test.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from env.cfj_env import WirelessJammingEnv, MAX_POWER_W

RESULTS_DIR = "results"
N_EPISODES  = 50          # per data point
N_RAND      = 1500        # random search budget per scene
SEED        = 42


def normal_wifi(env):
    env_score = []
    for ep in range(N_EPISODES):
        env.reset(seed=ep)
        d = env._pairwise_dist(env.user_positions, env.ap_positions)
        nearest = np.argmin(d, axis=1).astype(np.int32)
        env.association = nearest
        powers = np.zeros(env.num_aps, dtype=np.float32)
        powers[np.unique(nearest)] = MAX_POWER_W
        env_score.append(env.evaluate_policy(powers))
    return env_score


def smart_ap(env):
    s = []
    for ep in range(N_EPISODES):
        env.reset(seed=ep)
        powers = np.zeros(env.num_aps, dtype=np.float32)
        powers[np.unique(env.association)] = MAX_POWER_W
        s.append(env.evaluate_policy(powers))
    return s


def rl_proxy(env, seed_offset=0):
    s = []
    for ep in range(N_EPISODES):
        env.reset(seed=ep + seed_offset)
        rng = np.random.default_rng(ep + seed_offset + 1000)
        # start from smart-AP solution
        smart_powers = np.zeros(env.num_aps, dtype=np.float32)
        smart_powers[np.unique(env.association)] = MAX_POWER_W
        best_metrics = env.evaluate_policy(smart_powers)
        best_score   = best_metrics["sum_secrecy_capacity"]
        for _ in range(N_RAND):
            a = rng.uniform(-1, 1, size=env.num_aps).astype(np.float32)
            m = env.evaluate_policy(a)
            if m["sum_secrecy_capacity"] > best_score:
                best_score, best_metrics = m["sum_secrecy_capacity"], m
        s.append(best_metrics)
    return s


def aggregate(metric_list):
    return {
        "sec":   float(np.mean([m["sum_secrecy_capacity"] for m in metric_list])),
        "eve":   float(np.mean([m["sum_eve_capacity"]     for m in metric_list])),
        "ratio": float(np.mean([m["secrecy_ratio"]        for m in metric_list])) * 100,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)

    # ----- PLOT 1: secrecy vs. number of APs -----
    print("Plot 1: secrecy vs. AP count")
    ap_counts = [4, 5, 7, 9, 13]
    nw_y, sa_y, rl_y = [], [], []
    for n in ap_counts:
        env = WirelessJammingEnv(num_aps=n, num_users=2, num_eves=1,
                                 csi_noise_std=0.0, seed=SEED)
        nw = aggregate(normal_wifi(env))
        sa = aggregate(smart_ap(env))
        rl = aggregate(rl_proxy(env))
        nw_y.append(nw["sec"]); sa_y.append(sa["sec"]); rl_y.append(rl["sec"])
        print(f"  N={n:>2d}  normal={nw['sec']:.2f}  smart={sa['sec']:.2f}  rl={rl['sec']:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ap_counts, nw_y, "b-o", ms=7, label="Normal Wi-Fi")
    ax.plot(ap_counts, sa_y, "g-s", ms=7, label="Smart AP")
    ax.plot(ap_counts, rl_y, "r-^", ms=8, label="RL-CFJ proxy (perfect CSI)")
    ax.set_xlabel("Number of APs")
    ax.set_ylabel("Sum Secrecy Capacity (bps/Hz)")
    ax.set_title("Sum Secrecy Capacity vs. Number of APs")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot1_secrecy_vs_aps.png", dpi=150)
    plt.close()

    # ----- PLOT 2 & 3: secrecy & ratio vs. Eve CSI noise -----
    print("\nPlot 2/3: secrecy vs. Eve CSI noise (4 APs)")
    noise_levels = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    sec, ratio, eve = [], [], []
    for sigma in noise_levels:
        env = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                 csi_noise_std=sigma, seed=SEED)
        rl = aggregate(rl_proxy(env))
        sec.append(rl["sec"]); ratio.append(rl["ratio"]); eve.append(rl["eve"])
        print(f"  sigma={sigma:>4.1f}m  sec={rl['sec']:.2f}  ratio={rl['ratio']:.0f}%  "
              f"eve={rl['eve']:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(noise_levels, sec, "g-^", ms=7, linewidth=2, label="Sum secrecy")
    ax.plot(noise_levels, eve, "r--o", ms=6, linewidth=1.5, label="Sum Eve capacity")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.7, label="Perfect CSI")
    ax.set_xlabel(r"Eve location uncertainty $\sigma_\epsilon$ (m)")
    ax.set_ylabel("Capacity (bps/Hz)")
    ax.set_title("Effect of Imperfect Eve CSI on Secrecy and Eve Capacity")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot2_secrecy_vs_noise.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(noise_levels, ratio, "m-D", ms=7, linewidth=2)
    ax.set_xlabel(r"Eve location uncertainty $\sigma_\epsilon$ (m)")
    ax.set_ylabel("Secrecy Ratio (%)")
    ax.set_title("Effect of Imperfect Eve CSI on Secrecy Ratio")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot3_ratio_vs_noise.png", dpi=150)
    plt.close()

    # ----- PLOT 4: comparison bars -----
    print("\nPlot 4: bar comparison (4 APs, 2 users, 1 Eve)")
    env_clean = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                   csi_noise_std=0.0, seed=SEED)
    env_noisy = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                   csi_noise_std=5.0, seed=SEED)
    nw   = aggregate(normal_wifi(env_clean))
    sa   = aggregate(smart_ap(env_clean))
    rl_c = aggregate(rl_proxy(env_clean))
    rl_n = aggregate(rl_proxy(env_noisy))

    labels = ["Normal Wi-Fi", "Smart AP",
              "RL-CFJ\n(perfect CSI)", "RL-CFJ\n(noisy sigma=5m)"]
    sec_v = [nw["sec"],   sa["sec"],   rl_c["sec"],   rl_n["sec"]]
    eve_v = [nw["eve"],   sa["eve"],   rl_c["eve"],   rl_n["eve"]]
    rat_v = [nw["ratio"], sa["ratio"], rl_c["ratio"], rl_n["ratio"]]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, vals, ylabel, title, fmt in [
        (axes[0], sec_v, "Sum Secrecy Capacity (bps/Hz)",
         "Sum Secrecy Capacity", "{:.2f}"),
        (axes[1], eve_v, "Sum Eve Capacity (bps/Hz)",
         "Sum Eve Capacity (lower is better)", "{:.2f}"),
        (axes[2], rat_v, "Secrecy Ratio (%)",
         "Secrecy Ratio", "{:.0f}%"),
    ]:
        bars = ax.bar(x, vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if "Ratio" in title:
            ax.set_ylim(0, 110)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if "Ratio" in title else 0.05),
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
    plt.suptitle("Performance comparison (4 APs, 2 users, 1 Eve, "
                 f"{N_EPISODES} episodes)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/plot4_comparison_bar.png", dpi=150)
    plt.close()

    # Final printed table
    print("\nFinal comparison (proxy results):")
    print(f"{'Method':<28s} {'SecCap':>8s} {'EveCap':>8s} {'Ratio':>8s}")
    for name, m in [("Normal Wi-Fi", nw),
                    ("Smart AP", sa),
                    ("RL-CFJ proxy perfect", rl_c),
                    ("RL-CFJ proxy noisy s=5", rl_n)]:
        print(f"{name:<28s} {m['sec']:>8.2f} {m['eve']:>8.2f} "
              f"{m['ratio']:>7.0f}%")
    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

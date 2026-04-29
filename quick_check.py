"""
quick_check.py
==============

Tiny pipeline check (~30-60 seconds on CPU). Trains a single 2k-step SAC
agent at sigma=0, evaluates it against the three baselines, and prints
the comparison table. Use this to confirm the codebase runs end-to-end
before committing to a full `train.py + test.py` cycle.
"""

import os
import numpy as np
from stable_baselines3 import SAC

from env.cfj_env import WirelessJammingEnv, MAX_POWER_W

N_EPISODES = 50
TIMESTEPS  = 2_000


def evaluate_normal_wifi(env):
    sums = []
    for ep in range(N_EPISODES):
        env.reset(seed=ep)
        d = env._pairwise_dist(env.user_positions, env.ap_positions)
        nearest = np.argmin(d, axis=1)
        env.association = nearest.astype(np.int32)
        powers = np.zeros(env.num_aps, dtype=np.float32)
        powers[np.unique(nearest)] = MAX_POWER_W
        m = env.evaluate_policy(powers)
        sums.append(m["sum_secrecy_capacity"])
    return float(np.mean(sums))


def evaluate_smart_ap(env):
    sums = []
    for ep in range(N_EPISODES):
        env.reset(seed=ep)
        powers = np.zeros(env.num_aps, dtype=np.float32)
        powers[np.unique(env.association)] = MAX_POWER_W
        m = env.evaluate_policy(powers)
        sums.append(m["sum_secrecy_capacity"])
    return float(np.mean(sums))


def evaluate_rl(env, model):
    sums = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=ep)
        action, _ = model.predict(obs, deterministic=True)
        m = env.evaluate_policy(action)
        sums.append(m["sum_secrecy_capacity"])
    return float(np.mean(sums))


def main():
    print("Quick check: train 2k-step SAC, evaluate vs. baselines\n")
    env = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                             csi_noise_std=0.0, seed=42)

    nw = evaluate_normal_wifi(env)
    sa = evaluate_smart_ap(env)
    print(f"  Normal Wi-Fi    sum-secrecy = {nw:.2f}  bps/Hz")
    print(f"  Smart AP        sum-secrecy = {sa:.2f}  bps/Hz")

    print("\n  Training SAC for 2k steps...")
    model = SAC("MlpPolicy", env, verbose=0, seed=42,
                learning_rate=3e-4, buffer_size=10_000,
                batch_size=256, ent_coef="auto",
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=TIMESTEPS)

    rl = evaluate_rl(env, model)
    print(f"  RL-CFJ (2k SAC) sum-secrecy = {rl:.2f}  bps/Hz")

    print("\n  Sanity check: RL >= Smart AP >= Normal Wi-Fi expected")
    print(f"  RL/Smart ratio = {rl/sa:.2f},  Smart/Normal ratio = {sa/nw:.2f}")
    if rl > sa > nw:
        print("\n  PASS — pipeline works.\n")
    else:
        print("\n  WARN — ordering didn't hold. 2k steps may be too few; "
              "this is OK as a smoke test of the pipeline.\n")


if __name__ == "__main__":
    main()

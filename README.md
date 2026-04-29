# Cooperative Jamming for Physical Layer Security with Imperfect Eve CSI

This repository simulates Physical Layer Security (PLS) in a Wi-Fi network
controlled by a Soft Actor-Critic (SAC) deep-RL agent that allocates
transmit power across access points to maximise the **sum secrecy
capacity** of legitimate users in the presence of eavesdroppers.

## Baseline paper

This work re-implements the system model of:

> S. A. Hoseini, F. Bouhafs, N. Aboutorab, P. Sadeghi, and F. den Hartog,
> *"Cooperative Jamming for Physical Layer Security Enhancement Using Deep
> Reinforcement Learning"*, arXiv:2403.10342, 2024.

Equations (1)-(9) from the paper are implemented in `env/cfj_env.py` —
each is annotated in the docstrings.

## Project novelty

The paper assumes the network controller has **perfect** knowledge of every
eavesdropper's location.  In practice eavesdropper localisation is noisy
or stale.

This project extends the paper with **imperfect Eve channel state
information**: the RL agent observes Eve coordinates corrupted by
i.i.d. Gaussian noise of standard deviation `sigma_eps` metres.  The reward
is still computed from the *true* Eve position so the training signal
stays unbiased; this forces SAC to learn a power-allocation policy that
is robust to localisation error.

The contribution is implemented in three places:

1. `env/cfj_env.py` — `WirelessJammingEnv(..., csi_noise_std=sigma)` adds
   the noise inside `_build_observation()` only.
2. `train.py` — sweeps `NOISE_LEVELS = [0.0, 2.0, 5.0, 10.0]` m and saves
   one SAC model per noise level.
3. `test.py` — Plots 2 and 3 sweep secrecy capacity and secrecy ratio
   across `sigma_eps in {0, 1, 2, 3, 5, 7, 10} m`; Plot 4 adds a fourth
   bar for the imperfect-CSI policy.

## Repo layout

```
CY315/
  env/
    __init__.py
    cfj_env.py             # Gymnasium environment; eqs. (1)-(9) live here
  models/                  # auto-created; saved .zip SAC checkpoints
  results/                 # auto-created; output PNG plots
  results_obs_aware/       # plots from generate_plots_obs_aware.py
  train.py                 # train one SAC agent per Eve-CSI noise level
  test.py                  # generate the four comparison plots
  quick_check.py           # ~30 s end-to-end smoke test
  validation.py            # torch-free correctness checks (7 tests)
  generate_plots.py        # torch-free random-search proxy plots
  generate_plots_obs_aware.py  # torch-free CEM-based observation-aware proxy
  requirements.txt
  README.md
```

The two `generate_plots*` scripts exist so you can produce demo figures
**without installing torch**. They are NOT replacements for `train.py +
test.py` — they're stand-ins that approximate the SAC upper bound.
For real RL results, use `train.py` and `test.py`.

## Setup

```bash
python -m venv pls_env
# Windows:
pls_env\Scripts\activate
# macOS / Linux:
source pls_env/bin/activate

pip install -r requirements.txt
```

## How to run

### 0. Validate correctness (no torch needed, ~10 seconds)

```bash
python validation.py
```

Should print 7 PASS lines and "All validation checks passed." This
verifies the environment physics, paper equations, baseline distinctness,
and the CSI-noise isolation property — all without touching torch or SAC.

### 1. Smoke-test the pipeline (~30 s)

```bash
python quick_check.py
```

Should print three numbers (Normal Wi-Fi, Smart AP, 2k-step RL) and a
`PASS` line.

### 2. Train all SAC agents (~20-40 min on CPU)

```bash
python train.py
```

Default: 50 000 timesteps each at sigma in `{0.0, 2.0, 5.0, 10.0}` m.  Saves
`models/sac_noise_<sigma>.zip` and `results/training_convergence.png`.

Useful flags:

```bash
python train.py --timesteps 30000 --noise 0.0 5.0
python train.py --num-aps 7
```

### 3. Generate plots

```bash
python test.py --skip-plot1
```

Plot 1 (secrecy vs. number of APs) needs SAC models trained at each AP
count.  Use `--skip-plot1` if you only trained at the default 4 APs.
To enable Plot 1, train extra models like:

```bash
python train.py --num-aps 5 --noise 0.0
python train.py --num-aps 7 --noise 0.0
# ... save them as models/sac_naps_<N>_noise_0.0.zip (rename manually)
```

## How to read the results

| File                                  | What it shows                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------- |
| `training_convergence.png`            | Episode-return curves; each noise level should plateau                        |
| `plot1_secrecy_vs_aps.png`            | Trend from Fig. 3 of the paper: secrecy rises with AP density                 |
| `plot2_secrecy_vs_noise.png`          | **Novelty**: secrecy degrades smoothly as Eve-CSI noise increases             |
| `plot3_ratio_vs_noise.png`            | **Novelty**: secrecy ratio (% of users with positive secrecy) vs. noise       |
| `plot4_comparison_bar.png`            | Three-baseline + one-novelty bar chart for the 4-AP / 2-user / 1-Eve scene    |

The text table printed at the end of `python test.py` is what to copy
into the report.

## What was fixed relative to the previous iteration

- Folder/import mismatch (`env/` vs `envs/`) - now consistent.
- Smart-AP baseline now actually differs from Normal Wi-Fi (PLS-aware
  association vs. nearest-AP).
- No more silent fake-data fallback when a model is missing - missing
  models either skip a point on the plot or raise a clear error.
- Plot 1 only uses RL data when an RL model has actually been trained
  for that AP count.
- All seeds wired through (`numpy`, `random`, `torch`, `SAC`,
  `env.reset(seed=...)`).
- CSI noise verified to corrupt observation only, never reward.

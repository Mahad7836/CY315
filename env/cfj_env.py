"""
cfj_env.py
==========

Custom Gymnasium environment for Cooperative Friendly Jamming (CFJ) with
imperfect Channel State Information (CSI).

Faithful re-implementation of the system model from:
    Hoseini et al., "Cooperative Jamming for Physical Layer Security
    Enhancement Using Deep Reinforcement Learning", arXiv:2403.10342, 2024.

Novel contribution (this project):
    The agent observes Eve coordinates corrupted by Gaussian noise of
    standard deviation `csi_noise_std` (in metres). The TRUE Eve
    coordinates are used internally to compute the reward, so the agent
    must learn a power-allocation policy that is robust to localisation
    error in the eavesdropper.

Paper equation references in the docstrings below:
    eq. (1) Friis path-loss
    eq. (2) legitimate user Shannon capacity
    eq. (3) eavesdropper Shannon capacity (cooperative jamming)
    eq. (4) worst-case Eve = max over all Eves
    eq. (5) wiretap secrecy capacity = [C_legit - C_eve]+
    eq. (7) AP-selection rule (max secrecy under uniform max power)
    eq. (8) power-allocation optimisation problem
    eq. (9) reward = sum of positive secrecy capacities across users
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------- Physical / simulation constants from the paper ----------
MAP_SIZE_M       = 50.0          # 50 m x 50 m square area
FREQ_HZ          = 2.4e9         # Wi-Fi 2.4 GHz
SPEED_OF_LIGHT   = 3.0e8
WAVELENGTH_M     = SPEED_OF_LIGHT / FREQ_HZ
PATH_LOSS_EXP    = 2.0           # gamma = 2
NOISE_DBM        = -85.0         # noise floor at every node
NOISE_W          = 10 ** (NOISE_DBM / 10.0) * 1e-3  # ~3.16e-12 W
G_T              = 1.0           # unit antenna gains as in paper
G_R              = 1.0
MAX_POWER_W      = 1.0           # P_max per AP
BANDWIDTH_HZ     = 1.0           # capacities reported in bps/Hz


def friis_received_power(p_tx_w: np.ndarray, dist_m: np.ndarray) -> np.ndarray:
    """
    Eq. (1) — Friis transmission. Vectorised.
        p_r = p_t * G_t * G_r * (lambda / 4*pi)^2 * (1/d)^gamma

    Args:
        p_tx_w   : transmit power per AP, shape (N,)
        dist_m   : distances AP-to-receiver,  shape (N,) or (N, M)

    Returns:
        received power in Watts, broadcast to dist_m's shape.
    """
    # Avoid division by zero for receivers exactly co-located with an AP.
    safe_d = np.maximum(dist_m, 1e-3)
    free_space = (WAVELENGTH_M / (4.0 * np.pi)) ** 2
    if dist_m.ndim == 1:
        return p_tx_w * G_T * G_R * free_space * safe_d ** (-PATH_LOSS_EXP)
    # broadcast p_tx_w over the second axis
    return (p_tx_w[:, None] * G_T * G_R * free_space
            * safe_d ** (-PATH_LOSS_EXP))


class WirelessJammingEnv(gym.Env):
    """
    Gymnasium environment for cooperative friendly jamming with imperfect Eve CSI.

    State (observation):
        Concatenated, normalised positions of [APs, users, eavesdroppers].
        Eve positions in the *observation* are corrupted by N(0, sigma^2 * I_2)
        with sigma = `csi_noise_std`.  This is the novelty — the noise is in
        the observation only; the reward is computed from the true Eve
        positions so the RL signal is unbiased.

    Action:
        Continuous power vector P_t in [0, P_max]^N, one entry per AP.
        SAC outputs values in [-1, 1] which we map to [0, P_max].

    Reward:
        eq. (9) — sum over users of [C_user - C_eve_worst]+, computed from
        TRUE positions.

    Episode:
        One step per episode. Each reset randomises user / Eve placement,
        chooses AP-user association via eq. (7), then waits for a power
        action. (Single-step bandits are standard for this kind of
        snapshot-based power-allocation problem.)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_aps: int = 4,
        num_users: int = 2,
        num_eves: int = 1,
        csi_noise_std: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()

        if num_aps < 1 or num_users < 1 or num_eves < 1:
            raise ValueError("num_aps, num_users, num_eves must all be >= 1")

        self.num_aps    = int(num_aps)
        self.num_users  = int(num_users)
        self.num_eves   = int(num_eves)
        self.csi_noise_std = float(csi_noise_std)
        self.max_power  = MAX_POWER_W

        # Action space: continuous in [-1, 1]^N (SAC convention).
        # We map to [0, P_max] inside step().
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_aps,), dtype=np.float32
        )

        # Observation: 2D positions of APs, users, eves, normalised to [0, 1].
        obs_dim = 2 * (self.num_aps + self.num_users + self.num_eves)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Random number generator
        self._rng = np.random.default_rng(seed)

        # Place APs once on a coarse grid so they tile the map evenly.
        # (The paper places APs "sufficiently far from each other to cover
        # the entire map area"; a grid is the simplest reproducible choice.)
        self.ap_positions = self._tile_aps(self.num_aps)

        # Filled in by reset()
        self.user_positions: np.ndarray | None = None
        self.eve_positions:  np.ndarray | None = None
        self.association:    np.ndarray | None = None  # length num_users

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------
    def _tile_aps(self, n: int) -> np.ndarray:
        """Place APs on the smallest grid that fits n points, jittered slightly."""
        side = int(np.ceil(np.sqrt(n)))
        xs = np.linspace(MAP_SIZE_M * 0.1, MAP_SIZE_M * 0.9, side)
        ys = np.linspace(MAP_SIZE_M * 0.1, MAP_SIZE_M * 0.9, side)
        grid = np.array([(x, y) for x in xs for y in ys])[:n]
        # Small deterministic jitter so APs aren't perfectly axis-aligned
        rng = np.random.default_rng(42)
        grid = grid + rng.uniform(-1.0, 1.0, size=grid.shape)
        return np.clip(grid, 0.0, MAP_SIZE_M).astype(np.float32)

    def _random_positions(self, n: int) -> np.ndarray:
        return self._rng.uniform(0.0, MAP_SIZE_M, size=(n, 2)).astype(np.float32)

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Returns (len(a), len(b)) pairwise Euclidean distance matrix."""
        diff = a[:, None, :] - b[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    # ------------------------------------------------------------------
    # Capacity calculations (paper eqs. 2-5)
    # ------------------------------------------------------------------
    def _capacities(
        self,
        powers: np.ndarray,
        eve_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-user Shannon capacities (eq. 2) and per-AP worst-case Eve
        capacities (eqs. 3-4) for a given power vector and Eve placement.

        Args:
            powers        : (N,) AP transmit powers in Watts.
            eve_positions : (J, 2) Eve coordinates to use.

        Returns:
            C_user_per_ap : (N, K) Shannon capacity at user k from AP n.
            C_eve_worst   : (N,)   worst-case Eve capacity for traffic from AP n.
        """
        # Distances
        d_au = self._pairwise_dist(self.ap_positions, self.user_positions)  # (N, K)
        d_ae = self._pairwise_dist(self.ap_positions, eve_positions)        # (N, J)

        # Received power (Watts) at each user/Eve from each AP
        p_at_users = friis_received_power(powers, d_au)  # (N, K)
        p_at_eves  = friis_received_power(powers, d_ae)  # (N, J)

        # Total received power at each user / each Eve, summed over all APs
        total_user = p_at_users.sum(axis=0, keepdims=True)  # (1, K)
        total_eve  = p_at_eves .sum(axis=0, keepdims=True)  # (1, J)

        # Eq. (2): SINR at user k from AP n  =  p_n,k / (sum_{nu != n} p_nu,k + N)
        interf_user = total_user - p_at_users          # (N, K)
        sinr_user   = p_at_users / (interf_user + NOISE_W)
        C_user_per_ap = BANDWIDTH_HZ * np.log2(1.0 + sinr_user)  # (N, K)

        # Eq. (3): SINR at Eve j from AP n  =  p_n,j / (sum_{nu != n} p_nu,j + N)
        interf_eve = total_eve - p_at_eves             # (N, J)
        sinr_eve   = p_at_eves / (interf_eve + NOISE_W)
        C_eve      = BANDWIDTH_HZ * np.log2(1.0 + sinr_eve)      # (N, J)

        # Eq. (4): worst-case Eve = max over all Eves for each AP
        C_eve_worst = C_eve.max(axis=1)                # (N,)

        return C_user_per_ap, C_eve_worst

    def _select_associations(self) -> np.ndarray:
        """
        Eq. (7) — for each user k, pick the AP that maximises the secrecy
        capacity Cs(uk) under uniform max power. Uses TRUE Eve positions
        (the paper's controller knows them; the imperfect-CSI noise applies
        only to what the RL agent observes, not to the association rule).

        Returns:
            association : (K,) array of AP indices, one per user.
        """
        powers = np.full(self.num_aps, self.max_power, dtype=np.float32)
        C_user_per_ap, C_eve_worst = self._capacities(powers, self.eve_positions)
        # Secrecy if user k associates with AP n: C_user_per_ap[n,k] - C_eve_worst[n]
        secrecy = C_user_per_ap - C_eve_worst[:, None]    # (N, K)
        return np.argmax(secrecy, axis=0).astype(np.int32)  # (K,)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        """
        Concatenate AP/user/Eve positions, normalise to [0,1], and add Gaussian
        noise to the EVE coordinates only. Clip to [0,1] to stay inside the
        observation_space bounds.
        """
        # True Eve positions seen by the controller, plus Gaussian noise (eve only)
        eve_obs = self.eve_positions.copy()
        if self.csi_noise_std > 0.0:
            eve_obs = eve_obs + self._rng.normal(
                loc=0.0, scale=self.csi_noise_std, size=eve_obs.shape
            ).astype(np.float32)

        obs = np.concatenate(
            [self.ap_positions.ravel(),
             self.user_positions.ravel(),
             eve_obs.ravel()]
        ) / MAP_SIZE_M
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.user_positions = self._random_positions(self.num_users)
        self.eve_positions  = self._random_positions(self.num_eves)
        self.association    = self._select_associations()

        return self._build_observation(), {}

    def step(self, action: np.ndarray):
        # Map SAC's [-1, 1] action to [0, P_max]
        action = np.asarray(action, dtype=np.float32).flatten()
        powers = (action + 1.0) * 0.5 * self.max_power
        powers = np.clip(powers, 0.0, self.max_power)

        # Reward computed from TRUE Eve positions (eq. 9)
        sum_secrecy, info = self._evaluate(powers, true_eves=True)

        # Single-step episode
        terminated = True
        truncated  = False
        # Returning the same obs is conventional for one-shot episodes
        obs = self._build_observation()
        return obs, float(sum_secrecy), terminated, truncated, info

    # ------------------------------------------------------------------
    # Public evaluation helper (used by baselines and test.py)
    # ------------------------------------------------------------------
    def evaluate_policy(self, action_or_powers: np.ndarray) -> dict:
        """
        Score an arbitrary action / power vector against the current snapshot.
        If the input is in [-1, 1] (SAC action space) we map it; if it looks
        like Watts in [0, P_max] we use it directly.
        """
        arr = np.asarray(action_or_powers, dtype=np.float32).flatten()
        if arr.min() < 0.0:                       # SAC action space
            powers = (arr + 1.0) * 0.5 * self.max_power
        else:                                     # already Watts
            powers = arr.copy()
        powers = np.clip(powers, 0.0, self.max_power)
        _, info = self._evaluate(powers, true_eves=True)
        return info

    def _evaluate(
        self,
        powers: np.ndarray,
        true_eves: bool,
    ) -> tuple[float, dict]:
        """
        Inner evaluator returning (sum_secrecy, info_dict).
        `true_eves` is always True for reward — kept as a flag for future
        ablations that might score against noisy Eve.
        """
        eve_pos = self.eve_positions if true_eves else (
            self.eve_positions
            + self._rng.normal(0.0, self.csi_noise_std, self.eve_positions.shape)
        )
        C_user_per_ap, C_eve_worst = self._capacities(powers, eve_pos)

        # For each user, pick the capacity from its associated AP
        ap_idx = self.association                            # (K,)
        C_legit = C_user_per_ap[ap_idx, np.arange(self.num_users)]  # (K,)
        C_eve_at_user_ap = C_eve_worst[ap_idx]                      # (K,)

        per_user_secrecy = np.maximum(C_legit - C_eve_at_user_ap, 0.0)
        sum_secrecy      = float(per_user_secrecy.sum())
        secrecy_ratio    = float(np.mean(per_user_secrecy > 1e-9))
        sum_eve          = float(C_eve_worst.sum())

        info = {
            "sum_secrecy_capacity": sum_secrecy,
            "sum_eve_capacity":     sum_eve,
            "secrecy_ratio":        secrecy_ratio,
            "per_user_secrecy":     per_user_secrecy.tolist(),
            "powers_w":             powers.tolist(),
            "association":          ap_idx.tolist(),
        }
        return sum_secrecy, info


# Quick smoke test if run directly
if __name__ == "__main__":
    env = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                             csi_noise_std=0.0, seed=0)
    obs, _ = env.reset()
    print("obs shape:", obs.shape, "obs range:", obs.min(), obs.max())

    # Uniform max power (baseline) vs. all-off
    for label, action in [("max-power",  np.ones(env.num_aps, dtype=np.float32)),
                           ("all-off",    -np.ones(env.num_aps, dtype=np.float32))]:
        _, r, _, _, info = env.step(action)
        print(f"{label:>10s}  sum_secrecy={r:6.3f}  "
              f"sum_eve={info['sum_eve_capacity']:6.3f}  "
              f"ratio={info['secrecy_ratio']:.0%}")

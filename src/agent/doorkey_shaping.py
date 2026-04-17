"""
MiniGrid-DoorKey-8x8 · Q-Learning + Reward Shaping + Cosine Annealing
=======================================================================
Uso:
  python doorkey_8x8_shaping.py --mode sweep  [--sweep_count N]
  python doorkey_8x8_shaping.py --mode train  [--alpha A --gamma G --eps_end E --shaping_scale S --k K]
  python doorkey_8x8_shaping.py --mode test   (richiede q_table salvata su disco con --mode train)

Il flag --mode test è pienamente funzionante: la q_table viene salvata su disco
al termine del training e ricaricata dal test.
"""

import argparse
import math
import os
import pickle
import random
from collections import defaultdict, deque
from typing import cast

import gymnasium as gym
import minigrid  # noqa: F401 — registra gli ambienti MiniGrid
import numpy as np
import wandb
from gymnasium.spaces import Discrete
from gymnasium.wrappers import RecordVideo
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

# ──────────────────────────────────────────────────────────────────────────────
# Configurazione globale
# ──────────────────────────────────────────────────────────────────────────────
ENV_ID = "MiniGrid-DoorKey-8x8-v0"
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
SHAPING_SCALE = 0.5
N_EP = 50_000
N_EP_SWEEP = 10_000   # alzato da 4_000: più informativo su 8x8
SEED = 42
K = 1.0
Q_INIT = 5.0          # Optimistic Initial Values
QTABLE_PATH = "q_table_shaping.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# Sweep config
# ──────────────────────────────────────────────────────────────────────────────
sweep_config = {
    "method": "bayes",
    "metric": {"name": "performance/success_rate_100", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 2000, "eta": 2},
    "parameters": {
        "alpha":         {"values": [0.01, 0.15, 0.25]},
        "gamma":         {"values": [0.95, 0.99, 0.999]},
        "eps_end":       {"values": [0.01, 0.05]},
        "shaping_scale": {"values": [0.5, 1.0]},
        "k":             {"values": [0.5, 1.0, 2.0]},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Reward shaping basato su progresso BFS
# ──────────────────────────────────────────────────────────────────────────────
class DoorKeyProgressReward(gym.Wrapper):
    """Potential-based reward shaping F(s,s') = γ·Φ(s') − Φ(s).

    Φ(s) = progresso normalizzato verso il goal lungo il percorso
    ottimale: chiave → porta → goal.
    """

    def __init__(self, env: gym.Env, scale: float = 0.5, gamma: float = 0.99) -> None:
        super().__init__(env)
        self.scale = scale
        self.gamma = gamma
        self._d_total: int = 1
        self._prev_phi: float = 0.0
        self._key_pos: tuple[int, int] = (0, 0)
        self._door_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (0, 0)
        self._d_key_door: int = 0
        self._d_door_goal: int = 0

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = cast(MiniGridEnv, self.env.unwrapped)

        start: tuple[int, int] = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        self._key_pos = self._find("key")
        self._door_pos = self._find("door")
        self._goal_pos = self._find("goal")

        self._d_key_door = self._bfs(self._key_pos, self._door_pos, ignore_door=True)
        self._d_door_goal = self._bfs(self._door_pos, self._goal_pos, ignore_door=True)
        d_start_key = self._bfs(start, self._key_pos, ignore_door=False)
        self._d_total = max(d_start_key + self._d_key_door + self._d_door_goal + 2, 1)

        self._prev_phi = self._phi()
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        new_phi = self._phi()
        shaping = self.scale * (self.gamma * new_phi - self._prev_phi)
        self._prev_phi = new_phi
        info["raw_reward"] = float(env_reward)
        return obs, float(env_reward) + shaping, terminated, truncated, info

    # ------------------------------------------------------------------
    def _phi(self) -> float:
        u = cast(MiniGridEnv, self.env.unwrapped)
        agent: tuple[int, int] = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        has_key = u.carrying is not None and u.carrying.type == "key"
        door_obj = u.grid.get(*self._door_pos)
        door_open = door_obj is None or not getattr(door_obj, "is_locked", False)

        if not has_key and not door_open:
            d_rem = (
                self._bfs(agent, self._key_pos, ignore_door=False)
                + self._d_key_door
                + self._d_door_goal
                + 2
            )
        elif has_key and not door_open:
            d_rem = self._bfs(agent, self._door_pos, ignore_door=True) + self._d_door_goal + 1
        else:
            d_rem = self._bfs(agent, self._goal_pos, ignore_door=True)

        return float(np.clip(1.0 - d_rem / self._d_total, 0.0, 1.0))

    def _find(self, obj_type: str) -> tuple[int, int]:
        u = cast(MiniGridEnv, self.env.unwrapped)
        for x in range(u.width):
            for y in range(u.height):
                cell = u.grid.get(x, y)
                if cell is not None and cell.type == obj_type:
                    return (x, y)
        return (0, 0)

    def _bfs(
        self, start: tuple[int, int], goal: tuple[int, int], ignore_door: bool
    ) -> int:
        """BFS sul grid.

        ignore_door=True → le celle porta sono attraversabili (usato dopo
        aver preso la chiave o per stimare distanze pre-interazione).
        ignore_door=False → le porte bloccate sono ostacoli (usato per
        stimare il percorso reale quando la porta è ancora chiusa).
        """
        u = cast(MiniGridEnv, self.env.unwrapped)
        width, height = u.width, u.height

        if start == goal:
            return 0

        visited = np.zeros((width, height), dtype=bool)
        q: deque[tuple[int, int, int]] = deque()
        q.append((start[0], start[1], 0))
        visited[start[0], start[1]] = True

        while q:
            x, y, d = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if visited[nx, ny]:
                    continue
                cell = u.grid.get(nx, ny)
                if cell is not None and cell.type == "wall":
                    continue
                if (not ignore_door) and cell is not None and cell.type == "door" and getattr(cell, "is_locked", False):
                    continue
                if (nx, ny) == goal:
                    return d + 1
                visited[nx, ny] = True
                q.append((nx, ny, d + 1))

        return int(1e9)  # goal irraggiungibile


# ──────────────────────────────────────────────────────────────────────────────
# Utilità RL
# ──────────────────────────────────────────────────────────────────────────────
def make_env(
    env_id: str,
    shaping_scale: float,
    gamma: float,
    render_mode: str | None = None,
) -> gym.Env:
    base = gym.make(env_id, render_mode=render_mode)
    env = FullyObsWrapper(base)
    env = DoorKeyProgressReward(env, scale=shaping_scale, gamma=gamma)
    return env


def get_epsilon_cosine(
    episode: int,
    n_episodes: int,
    eps_start: float,
    eps_end: float,
    k: float = K,
) -> float:
    progress = episode / n_episodes
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * (progress ** k)))
    return eps_end + (eps_start - eps_end) * cosine_val


def safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def extract_state(obs) -> bytes:
    """Rappresentazione compatta: canali object-type + color (esclude stato)."""
    return obs["image"][:, :, [0, 2]].tobytes()


def make_q_table(n_actions: int):
    return defaultdict(lambda: np.full(n_actions, Q_INIT, dtype=np.float32))


def choose_action(state: bytes, q_table, epsilon: float, action_space) -> int:
    if random.random() < epsilon:
        return action_space.sample()
    return int(q_table[state].argmax())


def update_q(
    q_table,
    state: bytes,
    action: int,
    reward: float,
    next_state: bytes,
    alpha: float,
    gamma: float,
) -> float:
    td = (reward + gamma * float(q_table[next_state].max())) - q_table[state][action]
    q_table[state][action] += alpha * td
    return abs(td)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def run_loop(cfg, n_episodes: int):
    shaping_scale = getattr(cfg, "shaping_scale", SHAPING_SCALE)
    eps_end = getattr(cfg, "eps_end", EPS_END)
    k = getattr(cfg, "k", K)

    random.seed(SEED)
    np.random.seed(SEED)

    env = make_env(cfg.env_id, shaping_scale, cfg.gamma)
    env.reset(seed=SEED)

    n_actions = int(cast(Discrete, env.action_space).n)
    q_table = make_q_table(n_actions)
    success_history: list[float] = []
    visited_states: set[bytes] = set()

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = truncated = False
        ep_reward = ep_raw = 0.0
        steps = td_sum = 0.0

        epsilon = get_epsilon_cosine(episode, n_episodes, EPS_START, eps_end, k)

        while not (done or truncated):
            action = choose_action(state, q_table, epsilon, env.action_space)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = extract_state(next_obs)

            td = update_q(q_table, state, action, float(reward), next_state, cfg.alpha, cfg.gamma)
            td_sum += td
            visited_states.add(state)
            state = next_state
            ep_reward += float(reward)
            ep_raw += info.get("raw_reward", 0.0)
            steps += 1

        # Vittoria = episodio terminato con reward grezzo positivo (non timeout)
        success = 1.0 if (done and not truncated and ep_raw > 0) else 0.0
        success_history.append(success)
        sr = safe_mean(success_history[-100:])

        log = {
            "reward/shaped":                  ep_reward,
            "reward/raw":                     ep_raw,
            "performance/success_rate_100":   sr,
            "performance/episode_length":     steps,
            "training/mean_td_error":         td_sum / steps if steps > 0 else 0.0,
            "training/q_states_unique":       len(visited_states),
            "hyperparams/epsilon":            epsilon,
        }

        if episode % 500 == 0 or episode == n_episodes - 1:
            if q_table:
                keys = random.sample(list(q_table.keys()), min(1000, len(q_table)))
                vals = np.concatenate([q_table[kk] for kk in keys])
                log["training/mean_q_value"] = float(vals.mean())
                log["training/max_q_value"] = float(vals.max())
            print(
                f"[Ep {episode:>7}/{n_episodes}] "
                f"raw={ep_raw:.2f} shaped={ep_reward:.2f} | "
                f"eps={epsilon:.3f} | sr={sr:.2f} | "
                f"states={len(visited_states)}"
            )

        try:
            wandb.log(log)
        except Exception:
            print(f"[Ep {episode}] Run interrotta da Hyperband.")
            break

    env.close()
    return q_table


# ──────────────────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────────────────
def _sweep_train():
    """Chiamata dal wandb.agent durante lo sweep."""
    run = wandb.init(
        project="minigrid-qlearning",
        config={
            "env_id":        ENV_ID,
            "alpha":         ALPHA,
            "gamma":         GAMMA,
            "eps_start":     EPS_START,
            "eps_end":       EPS_END,
            "shaping_scale": SHAPING_SCALE,
            "k":             K,
            "n_episodes":    N_EP_SWEEP,
        },
    )
    run_loop(run.config, N_EP_SWEEP)
    wandb.finish()


def train_final(alpha: float, gamma: float, eps_end: float, shaping_scale: float, k: float):
    run = wandb.init(
        project="minigrid-qlearning",
        name="final-shaping",
        config={
            "env_id":        ENV_ID,
            "alpha":         alpha,
            "gamma":         gamma,
            "eps_start":     EPS_START,
            "eps_end":       eps_end,
            "shaping_scale": shaping_scale,
            "k":             k,
            "n_episodes":    N_EP,
            "run_type":      "final",
        },
    )
    q_table = run_loop(run.config, N_EP)
    wandb.finish()

    with open(QTABLE_PATH, "wb") as f:
        pickle.dump(dict(q_table), f)
    print(f"Q-table salvata in '{QTABLE_PATH}'")
    return q_table


def test_agent(q_table=None, n_runs: int = 5, video_folder: str = "./videos"):
    if q_table is None:
        if not os.path.exists(QTABLE_PATH):
            raise FileNotFoundError(
                f"'{QTABLE_PATH}' non trovato. Esegui prima --mode train."
            )
        with open(QTABLE_PATH, "rb") as f:
            raw = pickle.load(f)
        n_actions = int(cast(Discrete, gym.make(ENV_ID).action_space).n)
        q_table = make_q_table(n_actions)
        q_table.update(raw)
        print(f"Q-table caricata da '{QTABLE_PATH}' ({len(q_table)} stati)")

    print(f"\n--- FASE DI TEST ({n_runs} run) ---")
    os.makedirs(video_folder, exist_ok=True)

    env = make_env(ENV_ID, SHAPING_SCALE, GAMMA, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="doorkey-shaping-test",
        episode_trigger=lambda ep_id: True,
    )

    wins = 0
    for i in range(n_runs):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = truncated = False
        total_raw = 0.0

        while not (done or truncated):
            action = choose_action(state, q_table, epsilon=0.0, action_space=env.action_space)
            obs, _, done, truncated, info = env.step(action)
            state = extract_state(obs)
            total_raw += info.get("raw_reward", 0.0)

        if done and not truncated and total_raw > 0:
            wins += 1
            print(f"  Run {i+1}: VITTORIA ✓  (raw_reward={total_raw:.3f})")
        else:
            print(f"  Run {i+1}: fallimento ✗  (raw_reward={total_raw:.3f})")

    env.close()
    print(f"\nRisultato: {wins}/{n_runs} vittorie")
    print(f"Video salvati in '{video_folder}/'")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniGrid-DoorKey-8x8 · Q-Learning + Reward Shaping + Cosine Annealing"
    )
    parser.add_argument("--mode", choices=["sweep", "train", "test"], default="train")
    parser.add_argument("--sweep_count",   type=int,   default=15)
    parser.add_argument("--alpha",         type=float, default=ALPHA)
    parser.add_argument("--gamma",         type=float, default=GAMMA)
    parser.add_argument("--eps_end",       type=float, default=EPS_END)
    parser.add_argument("--shaping_scale", type=float, default=SHAPING_SCALE)
    parser.add_argument("--k",             type=float, default=K)
    parser.add_argument("--n_runs",        type=int,   default=5)
    args = parser.parse_args()

    if args.mode == "sweep":
        print(f"=== Sweep ({args.sweep_count} run × {N_EP_SWEEP} episodi) ===")
        sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning")
        wandb.agent(sweep_id, function=_sweep_train, count=args.sweep_count)
        print("Sweep completato. Controlla wandb.ai per i best hyperparameters.")

    elif args.mode == "train":
        print(
            f"=== Training finale ===\n"
            f"  alpha={args.alpha}, gamma={args.gamma}, eps_end={args.eps_end}, "
            f"shaping_scale={args.shaping_scale}, k={args.k}"
        )
        q_table = train_final(
            alpha=args.alpha,
            gamma=args.gamma,
            eps_end=args.eps_end,
            shaping_scale=args.shaping_scale,
            k=args.k,
        )
        print("Training completato! Avvio test...")
        test_agent(q_table, n_runs=args.n_runs)

    elif args.mode == "test":
        test_agent(n_runs=args.n_runs)

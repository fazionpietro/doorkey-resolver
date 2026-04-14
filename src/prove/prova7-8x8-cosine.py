import gymnasium as gym
from gymnasium.spaces import Discrete
import minigrid
import numpy as np
import random
import math
from collections import defaultdict, deque
from minigrid.wrappers import FullyObsWrapper
from minigrid.minigrid_env import MiniGridEnv
from typing import cast
from gymnasium.wrappers import RecordVideo
import wandb
import os
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Configurazione Globale
# ──────────────────────────────────────────────────────────────────────────────
ENV_ID = "MiniGrid-DoorKey-8x8-v0"
ALPHA = 0.1
GAMMA = 0.99

# Parametri Cosine Annealing
EPS_START = 1.0
EPS_END = 0.05

SHAPING_SCALE = 0.5
N_EP = 50_000
N_EP_SWEEP = 4_000
SEED = 42

K = 1

# ──────────────────────────────────────────────────────────────────────────────
# Sweep config (Aggiornato per Cosine Annealing)
# ──────────────────────────────────────────────────────────────────────────────
sweep_config = {
    "method": "bayes",
    "metric": {"name": "performance/success_rate_100", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 1000, "eta": 2},
    "parameters": {
        "alpha": {"values": [0.01, 0.15, 0.25]},
        "gamma": {"values": [0.95, 0.99, 0.999]},
        "eps_end": {"values": [0.01, 0.05]},
        "shaping_scale": {"values": [0.5, 1.0]},
        "k": {"values": [0.5, 1.0, 2.0]},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Reward Shaping
# ──────────────────────────────────────────────────────────────────────────────
class DoorKeyProgressReward(gym.Wrapper):
    def __init__(self, env, scale: float = 0.5, gamma: float = 0.99):
        super().__init__(env)
        self.scale = scale
        self._d_total = 1
        self._prev_progress = 0.0
        self._key_pos: tuple[int, int] = (0, 0)
        self._door_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (0, 0)
        self._d_key_door = 0
        self._d_door_goal = 0
        self.gamma = gamma

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = cast(MiniGridEnv, self.env.unwrapped)

        start: tuple[int, int] = (u.agent_pos[0], u.agent_pos[1])
        self._key_pos = self._find("key")
        self._door_pos = self._find("door")
        self._goal_pos = self._find("goal")
        self._d_key_door = self.bfs_steps(self._key_pos, self._door_pos)
        self._d_door_goal = self.bfs_steps(self._door_pos, self._goal_pos)
        self._d_total = (
            self.bfs_steps(start, self._key_pos) + self._d_key_door + self._d_door_goal
        )
        self._prev_progress = self._compute_progress()
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        new_progress = self._compute_progress()
        shaping = self.scale * (self.gamma * new_progress - self._prev_progress)
        self._prev_progress = new_progress

        info["raw_reward"] = float(env_reward)
        return obs, float(env_reward) + shaping, terminated, truncated, info

    def _compute_progress(self) -> float:
        u = cast(MiniGridEnv, self.env.unwrapped)
        agent: tuple[int, int] = (u.agent_pos[0], u.agent_pos[1])
        carrying = u.carrying
        has_key = carrying is not None and carrying.type == "key"
        door_obj = u.grid.get(*self._door_pos)
        door_open = door_obj is None or not getattr(door_obj, "is_locked", False)

        if not has_key and not door_open:
            d_rem = (
                self.bfs_steps(agent, self._key_pos)
                + self._d_key_door
                + self._d_door_goal
            )
        elif has_key and not door_open:
            d_rem = self.bfs_steps(agent, self._door_pos) + self._d_door_goal
        else:
            d_rem = self.bfs_steps(agent, self._goal_pos)

        return float(np.clip((self._d_total - d_rem) / self._d_total, 0.0, 1.0))

    def _find(self, obj_type: str) -> tuple[int, int]:
        u = cast(MiniGridEnv, self.env.unwrapped)
        for x in range(u.width):
            for y in range(u.height):
                cell = u.grid.get(x, y)
                if cell is not None and cell.type == obj_type:
                    return (x, y)
        return (0, 0)

    def bfs_steps(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        u = cast(MiniGridEnv, self.env.unwrapped)
        width, height = u.width, u.height

        if start == goal:
            return 0

        visited = np.zeros((width, height), dtype=bool)
        q = deque([(start[0], start[1], 0)])
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
                if (nx, ny) == goal:
                    return d + 1

                visited[nx, ny] = True
                q.append((nx, ny, d + 1))
        return int(1e9)


# ──────────────────────────────────────────────────────────────────────────────
# Utility & RL Core
# ──────────────────────────────────────────────────────────────────────────────
def make_env(
    env_id: str, shaping_scale: float, gamma: float, render_mode: str | None = None
) -> gym.Env:
    base_env = gym.make(env_id, render_mode=render_mode)
    env = FullyObsWrapper(base_env)
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
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (progress**k)))
    return eps_end + (eps_start - eps_end) * cosine_decay


def safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def extract_state(obs) -> bytes:
    image = obs["image"]
    return image[:, :, [0, 2]].tobytes()


def make_q_table(n_actions: int):
    # Inizializzazione a 5 per esplorazione ottimistica (Optimistic Initial Values)
    return defaultdict(lambda: np.full(n_actions, 5.0, dtype=np.float32))


def choose_action(state: bytes, q_table, epsilon: float, action_space) -> int:
    if random.random() < epsilon:
        return action_space.sample()
    return int(q_table[state].argmax())


def update_q_table(
    q_table,
    state: bytes,
    action: int,
    reward: float,
    next_state: bytes,
    alpha: float,
    gamma: float,
) -> float:
    current_q = q_table[state][action]
    max_next_q = float(q_table[next_state].max())
    td_error = (reward + gamma * max_next_q) - current_q
    q_table[state][action] = current_q + alpha * td_error
    return abs(td_error)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop condiviso
# ──────────────────────────────────────────────────────────────────────────────
def run_loop(cfg, n_episodes: int):
    shaping_scale = getattr(cfg, "shaping_scale", SHAPING_SCALE)

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
        done = False
        truncated = False

        ep_reward = 0.0
        ep_raw_reward = 0.0
        step_count = 0
        td_error_sum = 0.0

        # Cosine Annealing per l'Epsilon
        epsilon = get_epsilon_cosine(
            episode,
            n_episodes,
            EPS_START,
            getattr(cfg, "eps_end", EPS_END),
            k=getattr(cfg, "k", K),
        )

        while not (done or truncated):
            action = choose_action(state, q_table, epsilon, env.action_space)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = extract_state(next_obs)

            td_err = update_q_table(
                q_table, state, action, float(reward), next_state, cfg.alpha, cfg.gamma
            )
            td_error_sum += td_err

            visited_states.add(state)
            state = next_state

            ep_reward += float(reward)
            ep_raw_reward += info.get("raw_reward", 0.0)
            step_count += 1

        success = 1.0 if (done and not truncated and ep_raw_reward > 0) else 0.0
        success_history.append(success)
        success_rate = safe_mean(success_history[-100:])

        log_data = {
            "reward/raw": ep_raw_reward,
            "reward/shaped": ep_reward,
            "performance/success_rate_100": success_rate,
            "performance/episode_length": step_count,
            "training/mean_td_error": (
                td_error_sum / step_count if step_count > 0 else 0.0
            ),
            "training/q_states_unique": len(visited_states),
            "hyperparams/epsilon": epsilon,
        }

        if episode % 500 == 0 or episode == n_episodes - 1:
            if len(q_table) > 0:
                sample_keys = random.sample(
                    list(q_table.keys()), min(1000, len(q_table))
                )
                all_vals = np.concatenate([q_table[k] for k in sample_keys])
                log_data["training/mean_q_value"] = float(all_vals.mean())
                log_data["training/max_q_value"] = float(all_vals.max())

            print(
                f"[Ep {episode:>6}/{n_episodes}] "
                f"raw_rew={ep_raw_reward:.2f} | "
                f"eps={epsilon:.3f} | "
                f"success={success_rate:.2f} | "
                f"states={len(visited_states)}"
            )

        try:
            wandb.log(log_data)
        except Exception:
            print(f"[Ep {episode}] Run terminato da Hyperband.")
            break

    env.close()
    return q_table


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint sweep / train / test
# ──────────────────────────────────────────────────────────────────────────────
def train():
    run = wandb.init(
        project="minigrid-qlearning",
        config={
            "env_id": ENV_ID,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "shaping_scale": SHAPING_SCALE,
            "n_episodes": N_EP_SWEEP,
        },
    )
    run_loop(run.config, N_EP_SWEEP)
    wandb.finish()


def train_final(
    alpha: float,
    gamma: float,
    eps_end: float,
    shaping_scale: float = SHAPING_SCALE,
):
    run = wandb.init(
        project="minigrid-qlearning",
        name="final-training",
        config={
            "env_id": ENV_ID,
            "alpha": alpha,
            "gamma": gamma,
            "eps_start": EPS_START,
            "eps_end": eps_end,
            "shaping_scale": shaping_scale,
            "n_episodes": N_EP,
            "run_type": "final",
        },
    )
    q_table = run_loop(run.config, N_EP)
    wandb.finish()
    return q_table


def test_agent(q_table, n_runs: int = 5, video_folder: str = "./videos"):
    print(f"\n--- FASE DI TEST ({n_runs} run) ---")
    print(f"    Video salvati in: {video_folder}/")

    os.makedirs(video_folder, exist_ok=True)
    env = make_env(ENV_ID, SHAPING_SCALE, GAMMA, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="doorkey-test",
        episode_trigger=lambda ep_id: True,
    )

    wins = 0
    for i in range(n_runs):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action = choose_action(
                state, q_table, epsilon=0.0, action_space=env.action_space
            )
            obs, reward, done, truncated, _ = env.step(action)
            state = extract_state(obs)
            total_reward += float(reward)

        if done and not truncated and total_reward > 0:
            wins += 1
            print(f"  Run {i+1}: VITTORIA ✓  (reward={total_reward:.3f})")
        else:
            print(f"  Run {i+1}: fallimento ✗  (reward={total_reward:.3f})")

    env.close()
    print(f"\nRisultato: {wins}/{n_runs} vittorie")
    print(f"Video salvati in '{video_folder}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniGrid Q-Learning + W&B + Reward Shaping + Cosine Annealing"
    )

    # L'argomento --mode che causava l'errore è ora correttamente definito qui:
    parser.add_argument(
        "--mode",
        choices=["sweep", "train", "test"],
        default="train",
        help="Seleziona cosa eseguire: sweep, train o test.",
    )
    parser.add_argument(
        "--sweep_count", type=int, default=15, help="Numero di run per lo sweep."
    )

    # Iperparametri
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--eps_end", type=float, default=EPS_END)
    parser.add_argument("--shaping_scale", type=float, default=SHAPING_SCALE)
    parser.add_argument("--k", type=float, default=K)
    args = parser.parse_args()

    if args.mode == "sweep":
        print(f"=== Avvio Sweep ({args.sweep_count} run x {N_EP_SWEEP} episodi) ===")
        sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning")
        wandb.agent(sweep_id, function=train, count=args.sweep_count)
        print("\nSweep completato! Controlla wandb.ai per i migliori iperparametri.")
        print(
            "Poi esegui: python script.py --mode train --alpha X --gamma Y --eps_end Z ..."
        )

    elif args.mode == "train":
        print(
            f"=== Training finale ===\n"
            f"alpha={args.alpha}, gamma={args.gamma}, "
            f"eps_end={args.eps_end}, shaping_scale={args.shaping_scale}"
        )
        q_table = train_final(
            alpha=args.alpha,
            gamma=args.gamma,
            eps_end=args.eps_end,
            shaping_scale=args.shaping_scale,
        )
        print("Training completato! Avvio test...")
        test_agent(q_table, n_runs=5)

    elif args.mode == "test":
        print(
            "NOTA: Per eseguire solo il test dovresti salvare e caricare la q_table su disco (non implementato in questo script)."
        )
        print(
            "Esegui il comando con '--mode train' per addestrare e testare l'agente di seguito."
        )

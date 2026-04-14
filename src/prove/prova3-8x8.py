import gymnasium as gym
from gymnasium.spaces import Discrete
import minigrid
import numpy as np
import random
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
from minigrid.minigrid_env import MiniGridEnv
from typing import cast
from gymnasium.wrappers import RecordVideo
import wandb
from collections import deque

ENV_ID = "MiniGrid-DoorKey-8x8-v0"
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99995
SHAPING_SCALE = 0.5
N_EP = 50_000
N_EP_SWEEP = 15_000
WARMUP_FRAC = 0.10
DECAY_FRAC = 0.75
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Reward Shaping
# ──────────────────────────────────────────────────────────────────────────────


class DoorKeyProgressReward(gym.Wrapper):
    """
    Reward shaped basato sul progresso lungo la traiettoria:
        start → key → door → goal
    shaped_reward = env_reward + scale * (progress(s') - progress(s))

    Implementazione potential-based: garantisce che la policy ottimale
    non cambi rispetto all'ambiente originale.
    """

    def __init__(self, env, scale: float = 0.5):
        super().__init__(env)
        self.scale = scale
        self._d_total = 1
        self._prev_progress = 0.0
        self._key_pos: tuple[int, int] = (0, 0)
        self._door_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (0, 0)
        self._d_key_door = 0
        self._d_door_goal = 0

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
        shaping = self.scale * (new_progress - self._prev_progress)
        self._prev_progress = new_progress
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
        sx, sy = start
        q = deque([(start[0], start[1], 0)])
        visited[start[0], start[1]] = True

        while True:
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
# Sweep config
# ──────────────────────────────────────────────────────────────────────────────

sweep_config = {
    "method": "bayes",
    "metric": {"name": "success_rate_100", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 1000, "eta": 2},
    "parameters": {
        "alpha": {"distribution": "log_uniform_values", "min": 0.001, "max": 0.5},
        "gamma": {"values": [0.995, 0.999, 0.9999]},
        "eps_end": {"values": [0.05, 0.10]},
        "shaping_scale": {"values": [0.3, 0.5, 1.0]},
        "warmup_frac": {"values": [0.10, 0.12, 0.25]},
        "decay_frac": {"values": [0.60, 0.75, 0.85]},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────


def make_env(
    env_id: str, shaping_scale: float, render_mode: str | None = None
) -> gym.Env:
    """
    Ordine corretto dei wrapper:
        1. FullyObsWrapper  →  modifica le osservazioni
        2. DoorKeyProgressReward  →  modifica i reward (accede a unwrapped internamente)
    """
    base_env = gym.make(env_id, render_mode=render_mode)
    env = FullyObsWrapper(base_env)
    env = DoorKeyProgressReward(env, scale=shaping_scale)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────


def decrease_epsilon(
    episode: int,
    n_episodes: int,
    eps_start: float,
    eps_end: float,
    warmup_frac: float = WARMUP_FRAC,
    decay_frac: float = DECAY_FRAC,
) -> float:
    warmup_ep = int(n_episodes * warmup_frac)
    decay_ep = int(n_episodes * decay_frac)
    if episode < warmup_ep:
        return eps_start
    elif episode < warmup_ep + decay_ep:
        progress = (episode - warmup_ep) / decay_ep
        return eps_start - progress * (eps_start - eps_end)
    else:
        return eps_end


def safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def extract_state(obs) -> bytes:
    """
    Converte l'osservazione in una chiave compatta e veloce per la Q-table.
    Usa tobytes() invece di tuple() per ~10x speedup sui lookup del dizionario.
    Tiene solo i canali 0 (tipo oggetto) e 2 (stato oggetto), ignorando il colore.
    """
    image = obs["image"]
    return image[:, :, [0, 2]].tobytes()


def make_q_table(n_actions: int):
    return defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))


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
) -> None:
    current_q = q_table[state][action]
    max_next_q = float(q_table[next_state].max())
    q_table[state][action] = current_q + alpha * (
        reward + gamma * max_next_q - current_q
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop condiviso
# ──────────────────────────────────────────────────────────────────────────────


def run_loop(cfg, n_episodes: int):
    shaping_scale = getattr(cfg, "shaping_scale", SHAPING_SCALE)

    # Seed per riproducibilità
    random.seed(SEED)
    np.random.seed(SEED)

    env = make_env(cfg.env_id, shaping_scale)
    env.reset(seed=SEED)

    n_actions = int(cast(Discrete, env.action_space).n)
    q_table = make_q_table(n_actions)
    epsilon = EPS_START
    success_history: list[float] = []
    visited_states: set[bytes] = set()

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_raw_reward = 0.0  # reward senza shaping, per monitoraggio reale
        step_count = 0

        while not (done or truncated):
            action = choose_action(state, q_table, epsilon, env.action_space)
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = extract_state(next_obs)

            update_q_table(
                q_table, state, action, float(reward), next_state, cfg.alpha, cfg.gamma
            )
            visited_states.add(state)
            state = next_state
            ep_reward += float(reward)
            step_count += 1

        epsilon = decrease_epsilon(
            episode,
            n_episodes,
            EPS_START,
            getattr(cfg, "eps_end", EPS_END),
            warmup_frac=getattr(cfg, "warmup_frac", WARMUP_FRAC),
            decay_frac=getattr(cfg, "decay_frac", DECAY_FRAC),
        )

        # Successo = episodio terminato (goal raggiunto), non troncato (timeout)
        success = 1.0 if done and not truncated else 0.0
        success_history.append(success)
        success_rate = safe_mean(success_history[-100:])

        log_data = {
            "episode": episode,
            "episode_reward": ep_reward,
            "episode_length": step_count,
            "success": success,
            "success_rate_100": success_rate,
            "epsilon": epsilon,
            "q_pairs_visited": len(q_table) * n_actions,
            "q_states_unique": len(visited_states),
        }

        if episode % 200 == 0 or episode == n_episodes - 1:
            if len(q_table) > 0:
                sample_keys = random.sample(
                    list(q_table.keys()), min(1000, len(q_table))
                )
                all_vals = np.concatenate([q_table[k] for k in sample_keys])
                log_data["mean_q_value"] = float(all_vals.mean())
                log_data["max_q_value"] = float(all_vals.max())
            print(
                f"[Ep {episode:>6}/{n_episodes}] "
                f"reward={ep_reward:.3f} | "
                f"eps={epsilon:.3f} | "
                f"success_rate={success_rate:.2f} | "
                f"q_states={len(visited_states)}"
            )

        try:
            wandb.log(log_data, step=episode)
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
            "eps_decay": EPS_DECAY,
            "shaping_scale": SHAPING_SCALE,
            "n_episodes": N_EP_SWEEP,
            "warmup_frac": WARMUP_FRAC,
            "decay_frac": DECAY_FRAC,
        },
    )
    run_loop(run.config, N_EP_SWEEP)
    wandb.finish()


def train_final(
    alpha: float,
    gamma: float,
    eps_decay: float,
    eps_end: float,
    shaping_scale: float = SHAPING_SCALE,
    warmup_frac: float = WARMUP_FRAC,
    decay_frac: float = DECAY_FRAC,
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
            "eps_decay": eps_decay,
            "shaping_scale": shaping_scale,
            "n_episodes": N_EP,
            "run_type": "final",
            "warmup_frac": warmup_frac,
            "decay_frac": decay_frac,
        },
    )
    q_table = run_loop(run.config, N_EP)
    wandb.finish()
    return q_table


def test_agent(q_table, n_runs: int = 5, video_folder: str = "./videos"):
    print(f"\n--- FASE DI TEST ({n_runs} run) ---")
    print(f"    Video salvati in: {video_folder}/")

    # Stesso ordine di wrapper usato in training (senza shaping: scala 0)
    base_env = gym.make(ENV_ID, render_mode="rgb_array")
    env = FullyObsWrapper(base_env)
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

        if done and not truncated:
            wins += 1
            print(f"  Run {i+1}: VITTORIA ✓  (reward={total_reward:.3f})")
        else:
            print(f"  Run {i+1}: fallimento ✗  (reward={total_reward:.3f})")

    env.close()
    print(f"\nRisultato: {wins}/{n_runs} vittorie")
    print(f"Video salvati in '{video_folder}/'")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MiniGrid Q-Learning + W&B + Reward Shaping"
    )
    parser.add_argument("--mode", choices=["sweep", "train", "test"], default="train")
    parser.add_argument("--sweep_count", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--eps_decay", type=float, default=EPS_DECAY)
    parser.add_argument("--eps_end", type=float, default=EPS_END)
    parser.add_argument("--shaping_scale", type=float, default=SHAPING_SCALE)
    parser.add_argument("--warmup_frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--decay_frac", type=float, default=DECAY_FRAC)
    args = parser.parse_args()

    if args.mode == "sweep":
        print(f"=== Avvio Sweep ({args.sweep_count} run x {N_EP_SWEEP} episodi) ===")
        sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning")
        wandb.agent(sweep_id, function=train, count=args.sweep_count)
        print("\nSweep completato! Controlla wandb.ai per i migliori iperparametri.")
        print("Poi esegui: python script.py --mode train --alpha X --gamma Y ...")

    elif args.mode == "train":
        print(
            f"=== Training finale ===\n"
            f"alpha={args.alpha}, gamma={args.gamma}, "
            f"eps_decay={args.eps_decay}, eps_end={args.eps_end}, "
            f"shaping_scale={args.shaping_scale}"
        )
        q_table = train_final(
            alpha=args.alpha,
            gamma=args.gamma,
            eps_decay=args.eps_decay,
            eps_end=args.eps_end,
            shaping_scale=args.shaping_scale,
            warmup_frac=args.warmup_frac,
            decay_frac=args.decay_frac,
        )
        print("Training completato!")
        test_agent(q_table, n_runs=5)

    elif args.mode == "test":
        print("Esegui prima --mode train per addestrare e testare in sequenza.")

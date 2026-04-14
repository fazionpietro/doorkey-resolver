import gymnasium as gym
from gymnasium.spaces import Discrete
import minigrid
import numpy as np
import random
import math
from collections import defaultdict
from minigrid.wrappers import FullyObsWrapper
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
EPS_START = 1.0
EPS_END = 0.05
SHAPING_SCALE = 0.5
N_EP = 70_000
N_EP_SWEEP = 15_000
SEED = 42
WARMUP_FRAC = 0.10

# ──────────────────────────────────────────────────────────────────────────────
# Sweep config
# ──────────────────────────────────────────────────────────────────────────────
sweep_config = {
    "method": "bayes",
    "metric": {"name": "performance/success_rate_100", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 1000, "eta": 2},
    "parameters": {
        "alpha": {"values": [0.01, 0.15, 0.30]},
        "gamma": {"values": [0.95, 0.99, 0.999]},
        "eps_end": {"values": [0.01, 0.05]},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility & RL Core
# ──────────────────────────────────────────────────────────────────────────────
def make_env(env_id: str, render_mode: str | None = None) -> gym.Env:
    base_env = gym.make(env_id, render_mode=render_mode)
    return FullyObsWrapper(base_env)


def get_epsilon_cosine(
    episode: int,
    n_episodes: int,
    eps_start: float,
    eps_end: float,
    warmup_frac: float = WARMUP_FRAC,
) -> float:
    warmup_ep = int(n_episodes * warmup_frac)
    exploit_ep = warmup_ep
    decay_start = warmup_ep
    decay_end = n_episodes - exploit_ep

    if episode < decay_start:
        return eps_start
    elif episode >= decay_end:
        return eps_end
    else:
        progress = (episode - decay_start) / (decay_end - decay_start)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return eps_end + (eps_start - eps_end) * cosine_decay


def safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def extract_state(obs) -> bytes:
    image = obs["image"]
    return image[:, :, [0, 2]].tobytes()


def make_q_table(n_actions: int):
    return defaultdict(lambda: np.full(n_actions, 2.0, dtype=np.float32))


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
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def run_loop(cfg, n_episodes: int):
    random.seed(SEED)
    np.random.seed(SEED)

    env = make_env(cfg.env_id)
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
        step_count = 0
        td_error_sum = 0.0

        epsilon = get_epsilon_cosine(
            episode,
            n_episodes,
            EPS_START,
            getattr(cfg, "eps_end", EPS_END),
            warmup_frac=getattr(cfg, "warmup_frac", WARMUP_FRAC),
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
            step_count += 1

        success = 1.0 if (done and not truncated and ep_reward > 0) else 0.0
        success_history.append(success)
        success_rate = safe_mean(success_history[-100:])

        log_data = {
            "reward/episode": ep_reward,
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
                f"[Ep {episode:>7}/{n_episodes}] "
                f"reward={ep_reward:.2f} | "
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
# Entrypoint
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
            "n_episodes": N_EP_SWEEP,
        },
    )
    run_loop(run.config, N_EP_SWEEP)
    wandb.finish()


def train_final(alpha: float, gamma: float, eps_end: float):
    run = wandb.init(
        project="minigrid-qlearning",
        name="final-training",
        config={
            "env_id": ENV_ID,
            "alpha": alpha,
            "gamma": gamma,
            "eps_start": EPS_START,
            "eps_end": eps_end,
            "n_episodes": N_EP,
            "run_type": "final",
        },
    )
    q_table = run_loop(run.config, N_EP)
    wandb.finish()
    return q_table


def test_agent(q_table, n_runs: int = 5, video_folder: str = "./videos"):
    print(f"\n--- FASE DI TEST ({n_runs} run) ---")
    os.makedirs(video_folder, exist_ok=True)

    env = make_env(ENV_ID, render_mode="rgb_array")
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
    parser = argparse.ArgumentParser(description="MiniGrid Q-Learning 16x16")
    parser.add_argument("--mode", choices=["sweep", "train", "test"], default="train")
    parser.add_argument("--sweep_count", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--eps_end", type=float, default=EPS_END)
    args = parser.parse_args()

    if args.mode == "sweep":
        print(f"=== Avvio Sweep ({args.sweep_count} run x {N_EP_SWEEP} episodi) ===")
        sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning")
        wandb.agent(sweep_id, function=train, count=args.sweep_count)

    elif args.mode == "train":
        print(
            f"=== Training finale === alpha={args.alpha} gamma={args.gamma} eps_end={args.eps_end}"
        )
        q_table = train_final(alpha=args.alpha, gamma=args.gamma, eps_end=args.eps_end)
        print("Training completato!")
        test_agent(q_table, n_runs=5)

    elif args.mode == "test":
        print("Esegui prima --mode train.")

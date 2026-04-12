import gymnasium as gym
import minigrid
import numpy as np
import random
from collections import defaultdict
import time
from minigrid.wrappers import FullyObsWrapper
from minigrid.minigrid_env import MiniGridEnv
from typing import cast
from gymnasium.wrappers import RecordVideo
import wandb

ENV_ID = "MiniGrid-DoorKey-8x8-v0"
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99995
SHAPING_SCALE = 0.5
N_EP = 50_000
N_EP_SWEEP = 25_000
WARMUP_FRAC = 0.10
DECAY_FRAC = 0.75

# ──────────────────────────────────────────────────────────────────────────────
# Reward Shaping
# ──────────────────────────────────────────────────────────────────────────────


class DoorKeyProgressReward(gym.Wrapper):
    """
    Reward shaped basato sul progresso lungo la traiettoria:
        start → key → door → goal
    shaped_reward = env_reward + scale * (progress(s') - progress(s))
    """

    def __init__(self, env, scale: float = 0.5):
        super().__init__(env)
        self.scale = scale
        self._d_total = 1
        self._prev_progress = 0.0
        self._key_pos: np.ndarray = np.zeros(2, dtype=int)
        self._door_pos: np.ndarray = np.zeros(2, dtype=int)
        self._goal_pos: np.ndarray = np.zeros(2, dtype=int)
        self._d_key_door = 0
        self._d_door_goal = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        u = cast(MiniGridEnv, self.env.unwrapped)

        start = np.array(u.agent_pos)
        self._key_pos = self._find("key")
        self._door_pos = self._find("door")
        self._goal_pos = self._find("goal")

        d_start_key = self._mdist(start, self._key_pos)
        self._d_key_door = self._mdist(self._key_pos, self._door_pos)
        self._d_door_goal = self._mdist(self._door_pos, self._goal_pos)
        self._d_total = d_start_key + self._d_key_door + self._d_door_goal or 1

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
        agent = np.array(u.agent_pos)
        carrying = u.carrying
        has_key = carrying is not None and carrying.type == "key"
        door_obj = u.grid.get(*self._door_pos)
        door_open = door_obj is None or not getattr(door_obj, "is_locked", False)

        if not has_key and not door_open:
            d_rem = (
                self._mdist(agent, self._key_pos) + self._d_key_door + self._d_door_goal
            )
        elif has_key and not door_open:
            d_rem = self._mdist(agent, self._door_pos) + self._d_door_goal
        else:
            d_rem = self._mdist(agent, self._goal_pos)

        return float(np.clip((self._d_total - d_rem) / self._d_total, 0.0, 1.0))

    def _find(self, obj_type: str) -> np.ndarray:
        u = cast(MiniGridEnv, self.env.unwrapped)
        for x in range(u.width):
            for y in range(u.height):
                cell = u.grid.get(x, y)
                if cell is not None and cell.type == obj_type:
                    return np.array([x, y])
        return np.zeros(2, dtype=int)

    @staticmethod
    def _mdist(a: np.ndarray, b: np.ndarray) -> int:
        return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Sweep config
# ──────────────────────────────────────────────────────────────────────────────

sweep_config = {
    "method": "bayes",
    "metric": {"name": "success_rate_100", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 1000, "eta": 2},
    "parameters": {
        "alpha": {"distribution": "log_uniform_values", "min": 0.001, "max": 0.5},
        "gamma": {"values": [0.95, 0.99, 0.999]},
        "eps_end": {"values": [0.05, 0.10]},
        "shaping_scale": {"values": [0.3, 0.5, 1.0]},
        "warmup_frac": {"values": [0.05, 0.10, 0.20]},
        "decay_frac": {"values": [0.60, 0.75, 0.85]},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def decrease_epsilon(
    episode,
    n_episodes,
    eps_start,
    eps_end,
    warmup_frac=WARMUP_FRAC,
    decay_frac=DECAY_FRAC,
):
    warmup_ep = int(n_episodes * warmup_frac)
    decay_ep = int(n_episodes * decay_frac)
    if episode < warmup_ep:
        return eps_start
    elif episode < warmup_ep + decay_ep:
        progress = (episode - warmup_ep) / decay_ep
        return eps_start - progress * (eps_start - eps_end)
    else:
        return eps_end


def safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0


def extract_state(obs):
    # Gestisce sia dict {"image": arr} sia array diretto
    if isinstance(obs, dict):
        image = obs["image"]
    else:
        image = obs  # array diretto

    assert (
        image.ndim == 3 and image.shape[2] >= 3
    ), f"Formato inatteso: shape={image.shape}"
    return (tuple(image[:, :, 0].flatten()), tuple(image[:, :, 2].flatten()))


def choose_action(state, q_table, epsilon, action_space):
    if random.uniform(0, 1) < epsilon:
        return action_space.sample()
    q_vals = np.array([q_table[(state, a)] for a in range(action_space.n)])
    return int(q_vals.argmax())


def update_q_table(
    q_table, state, action, reward, next_state, alpha, gamma, action_space
):
    current_q = q_table[(state, action)]
    max_next_q = max(q_table[(next_state, a)] for a in range(action_space.n))
    q_table[(state, action)] = current_q + alpha * (
        reward + gamma * max_next_q - current_q
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training loop condiviso
# ──────────────────────────────────────────────────────────────────────────────


def run_loop(cfg, n_episodes):
    shaping_scale = getattr(cfg, "shaping_scale", SHAPING_SCALE)

    base_env = gym.make(cfg.env_id)
    env = DoorKeyProgressReward(base_env, scale=shaping_scale)
    env = FullyObsWrapper(env)

    q_table = defaultdict(lambda: 0.0)
    epsilon = cfg.eps_start
    success_history = []
    visited_states = set()
    last_q_metrics = {}

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        ep_reward = 0.0
        step_count = 0

        while not (done or truncated):
            action = choose_action(state, q_table, epsilon, env.action_space)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = extract_state(next_obs)

            update_q_table(
                q_table,
                state,
                action,
                reward,
                next_state,
                cfg.alpha,
                cfg.gamma,
                env.action_space,
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
            "q_pairs_visited": len(q_table),
            "q_states_unique": len(visited_states),
        }

        if (episode % 200 == 0 or episode == n_episodes - 1) and len(q_table) > 0:
            sample = random.sample(list(q_table.values()), min(1000, len(q_table)))
            log_data["mean_q_value"] = float(np.mean(sample))
            log_data["max_q_value"] = float(np.max(sample))
            log_data["q_states_unique"] = len(visited_states)
            log_data["q_pairs_visited"] = len(q_table)
            print(
                f"[Ep {episode:>6}/{n_episodes}] "
                f"reward={ep_reward:.3f} | "
                f"eps={epsilon:.3f} | "
                f"success_rate={success_rate:.2f} | "
                f"q_states={log_data['q_states_unique']}"
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


def train_final(alpha, gamma, eps_decay, eps_end, shaping_scale=SHAPING_SCALE):
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
            "warmup_frac": WARMUP_FRAC,
            "decay_frac": DECAY_FRAC,
        },
    )
    q_table = run_loop(run.config, N_EP)
    wandb.finish()
    return q_table


def test_agent(q_table, n_runs=5, video_folder="./videos"):
    print(f"\n--- FASE DI TEST ({n_runs} run) ---")
    print(f"    Video salvati in: {video_folder}/")

    # render_mode="rgb_array" obbligatorio per RecordVideo
    base_env = gym.make(ENV_ID, render_mode="rgb_array")
    base_env = FullyObsWrapper(base_env)
    env = RecordVideo(
        base_env,
        video_folder=video_folder,
        name_prefix="doorkey-test",
        episode_trigger=lambda ep_id: True,  # registra tutti gli episodi
    )

    wins = 0

    for i in range(n_runs):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        reward = 0.0

        while not (done or truncated):
            action = choose_action(
                state, q_table, epsilon=0.0, action_space=env.action_space
            )
            obs, reward, done, truncated, _ = env.step(action)
            state = extract_state(obs)

        if float(reward) > 0:
            wins += 1
            print(f"  Run {i+1}: VITTORIA ✓")
        else:
            print(f"  Run {i+1}: fallimento ✗")

    env.close()  # ← obbligatorio: finalizza e scrive i file .mp4
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
        )
        print("Training completato!")
        test_agent(q_table, n_runs=5)

    elif args.mode == "test":
        print("Esegui prima --mode train per addestrare e testare in sequenza.")

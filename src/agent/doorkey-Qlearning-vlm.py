#!/usr/bin/env python3
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import time
from collections import defaultdict, deque
import argparse
import math
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import cast
from gymnasium.spaces import Discrete
import wandb

from env.factory import make_env
from env.rewardsystem import RewardConfig, DoorKeyRewardSystem
from env import doorkey_events as doorev
from env.VLMDoorkeyWrapper import MoondreamDenseRewardWrapper

SEED = 42

DEFAULT_ENV_ID = "MiniGrid-DoorKey-6x6-v0"


class StateEncoder:
    def encode(self, env, info=None):
        """
        Estrae uno stato discreto e hashabile dall'ambiente MiniGrid.
        """
        # Accediamo all'ambiente base aggirando i wrapper
        unwrapped_env = env.unwrapped

        # 1. Posizione (x, y) e direzione (0-3) dell'agente
        agent_pos = unwrapped_env.agent_pos
        agent_dir = unwrapped_env.agent_dir

        # 2. L'agente ha raccolto la chiave?
        carrying = unwrapped_env.carrying
        has_key = getattr(carrying, "type", None) == "key"

        # 3. Qual è lo stato della porta? (Aperta o chiusa)
        # Cerchiamo la porta nella griglia
        door_open = False
        for i in range(unwrapped_env.grid.width):
            for j in range(unwrapped_env.grid.height):
                cell = unwrapped_env.grid.get(i, j)
                if cell is not None and cell.type == "door":
                    door_open = cell.is_open
                    break  # In DoorKey c'è tipicamente una sola porta

        # Restituiamo una tupla che il Q-Learning può usare come chiave del dizionario
        return (agent_pos[0], agent_pos[1], agent_dir, has_key, door_open)


class QLearningAgent:
    def __init__(
        self,
        n_actions=7,
        alpha=0.15,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.q[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)

        return int(np.random.choice(best_actions))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.q[s_next])
        td_target = r + self.gamma * best_next
        td_error = td_target - self.q[s][a]
        self.q[s][a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Trainer:
    def __init__(self, env, agent, encoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder

    def train(self, episodes=5000, max_steps=300, log_every=100):
        rewards = []
        avg_rewards = []
        success_buffer = deque(maxlen=100)

        for ep in range(episodes):

            if ep == 0:
                obs, info = self.env.reset(seed=SEED)
            else:
                obs, info = self.env.reset()

            # Sostituisci la riga 91 con:
            state = self.encoder.encode(self.env, info)
            ep_reward = 0.0
            info_next = info
            steps_taken = 0

            for step in range(max_steps):
                action = self.agent.act(state)
                obs_next, reward, terminated, truncated, info_next = self.env.step(
                    action
                )
                done = terminated or truncated

                next_state = self.encoder.encode(self.env, info_next)
                self.agent.update(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward
                steps_taken += 1

                if done:
                    break

            self.agent.decay_epsilon()

            rewards.append(ep_reward)

            # Estrazione metriche per W&B
            final_stage = info_next.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value

            is_success = 1 if final_stage == "goal_reached" else 0
            success_buffer.append(is_success)
            current_success_rate = np.mean(success_buffer)

            stage_map = {
                "no_key": 0,
                "has_key": 1,
                "door_open": 2,
                "goal_reached": 3,
                "unknown": -1,
            }

            # Logging su WandB ad ogni episodio
            wandb.log(
                {
                    "train/episode": ep,
                    "train/reward": ep_reward,
                    "train/steps": steps_taken,
                    "train/epsilon": self.agent.epsilon,
                    "train/success": is_success,
                    "train/success_rate_100ep": current_success_rate,
                    "train/final_stage_idx": stage_map.get(final_stage, -1),
                }
            )

            if ep % log_every == 0:
                avg = np.mean(rewards[-log_every:])
                avg_rewards.append(avg)
                print(
                    f"Ep {ep:5d}: reward={ep_reward:.2f}, avg_100={avg:.2f}, "
                    f"succ_rate={current_success_rate:.2f}, ε={self.agent.epsilon:.3f}, stage={final_stage}"
                )

        return rewards, avg_rewards

    def evaluate(self, episodes=100, max_steps=300):
        rewards = []
        successes = 0

        for _ in range(episodes):
            obs, info = self.env.reset()
            state = self.encoder.encode(self.env, info)
            ep_reward = 0.0
            info_next = info

            for _ in range(max_steps):
                action = self.agent.act(state)
                obs, reward, terminated, truncated, info_next = self.env.step(action)
                state = self.encoder.encode(self.env, info_next)
                ep_reward += reward
                if terminated or truncated:
                    break

            rewards.append(ep_reward)

            final_stage = info_next.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value
            if final_stage == "goal_reached":
                successes += 1

        avg_reward = np.mean(rewards)
        success_rate = successes / episodes

        # Log evaluation metrics
        wandb.log({"eval/avg_reward": avg_reward, "eval/success_rate": success_rate})

        return avg_reward, success_rate


def train_sweep():
    """Questa è la funzione che WandB chiamerà per ogni run dello sweep"""
    wandb.init()
    config = wandb.config

    print(
        f"Avvio run c eon: alpha={config.alpha:.3f}, gamma={config.gamma:.3f}, eps_decay={config.eps_decay:.4f}"
    )

    # Creazione ambiente
    env = gym.make(DEFAULT_ENV_ID, render_mode="rgb_array")
    env = MoondreamDenseRewardWrapper(env, beta=0.2, query_every=10)
    n_actions = int(cast(Discrete, env.action_space).n)

    # Inizializza le componenti con i parametri dello SWEEP
    encoder = StateEncoder()
    agent = QLearningAgent(
        n_actions=n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_decay=config.eps_decay,
    )
    trainer = Trainer(env, agent, encoder)

    trainer.train(episodes=3000, log_every=500)

    eval_reward, eval_success = trainer.evaluate(episodes=50)

    wandb.log({"sweep/final_success_rate": eval_success})

    print(f"Run conclusa. Success Rate: {eval_success*100:.1f}%")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="DoorKey Q-Learning con WandB. Usa --mode per scegliere tra training singolo e sweep."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "sweep"],
        default="train",
        help="'train' per una singola run, 'sweep' per avviare un WandB sweep agent (default: train)",
    )
    # Parametri per la singola run
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--project_name", type=str, default="doorkey-qlearning")
    # Parametri per lo sweep
    parser.add_argument(
        "--sweep_count",
        type=int,
        default=20,
        help="Numero di run da eseguire nello sweep (default: 20).",
    )
    args = parser.parse_args()

    if args.mode == "sweep":
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "eval/success_rate", "goal": "maximize"},
            "parameters": {
                "alpha": {"min": 0.05, "max": 0.5},
                "gamma": {"min": 0.90, "max": 0.999},
                "eps_decay": {"min": 0.990, "max": 0.9995},
            },
        }
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        print(
            f"Sweep creato automaticamente: id='{sweep_id}' | count={args.sweep_count}"
        )
        wandb.agent(sweep_id, function=train_sweep, count=args.sweep_count)
        return

    wandb.init(
        project=args.project_name,
        name=f"Run_ep{args.episodes}_alpha{args.alpha}",
        config={
            "episodes": args.episodes,
            "learning_rate": args.alpha,
            "gamma": args.gamma,
            "epsilon_decay": args.eps_decay,
            "env_id": "MiniGrid-DoorKey-6x6-v0",
            "agent_type": "QLearning_RelativeEncoder",
        },
    )

    print(f"Creazione ambiente DoorKey 6x6...")
    env = gym.make(DEFAULT_ENV_ID, render_mode="rgb_array")
    env = MoondreamDenseRewardWrapper(env, beta=0.2, query_every=10)
    n_actions = int(cast(Discrete, env.action_space).n)

    encoder = StateEncoder()
    agent = QLearningAgent(
        n_actions=n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.eps_decay,
    )
    trainer = Trainer(env, agent, encoder)

    print(f"Training avviato per {args.episodes} episodi...")
    rewards, avg_rewards = trainer.train(episodes=args.episodes)

    print("Valutazione...")
    eval_reward, eval_success = trainer.evaluate()
    print(f"Evaluation reward medio: {eval_reward:.2f}")
    print(f"Evaluation success rate: {eval_success*100:.1f}%")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()

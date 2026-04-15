#!/usr/bin/env python3
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
from env.rewardsystem import RewardConfig
from env import doorkey_events as doorev


class StateEncoder:
    """Codifica lo stato in base al target dello Stage e aggiunge un radar locale per i muri"""

    def encode(self, env, info):
        base = env.unwrapped
        ax, ay = base.agent_pos
        d = base.agent_dir

        stage = env.get_wrapper_attr("curr_stage")
        stage_name = stage.value if stage is not None else "no_key"

        if stage_name == "no_key":
            target_pos = env.get_wrapper_attr("key_pos")
            tx, ty = target_pos if target_pos is not None else (ax, ay)
        elif stage_name == "has_key":
            target_pos = env.get_wrapper_attr("door_pos")
            tx, ty = target_pos if target_pos is not None else (ax, ay)
        elif stage_name == "door_open":
            target_pos = env.get_wrapper_attr("goal_pos")
            tx, ty = target_pos if target_pos is not None else (ax, ay)
        else:
            tx, ty = ax, ay

        dx = tx - ax
        dy = ty - ay

        stage_map = {"no_key": 0, "has_key": 1, "door_open": 2, "goal_reached": 3}
        stage_idx = stage_map.get(stage_name, 0)

        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        fwd_vec = dirs[d]
        left_vec = dirs[(d - 1) % 4]
        right_vec = dirs[(d + 1) % 4]

        def is_wall(dx_vec, dy_vec):
            nx, ny = ax + dx_vec, ay + dy_vec
            # Controlliamo che le coordinate non escano dalla griglia
            if 0 <= nx < base.width and 0 <= ny < base.height:
                cell = base.grid.get(nx, ny)
                # Consideriamo ostacolo solo i muri veri e propri (non le porte chiuse)
                return 1 if cell is not None and cell.type == "wall" else 0
            return 1  # I bordi del livello sono considerati muri

        wall_front = is_wall(*fwd_vec)
        wall_left = is_wall(*left_vec)
        wall_right = is_wall(*right_vec)

        # 3. LO STATO FINALE DIVENTA PIÙ RICCO
        return (dx, dy, d, stage_idx, wall_front, wall_left, wall_right)


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

    def act(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
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
            obs, info = self.env.reset()
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

    def evaluate(self, episodes=50, max_steps=300):
        rewards = []
        successes = 0

        for _ in range(episodes):
            obs, info = self.env.reset()
            state = self.encoder.encode(self.env, info)
            ep_reward = 0.0
            info_next = info

            for _ in range(max_steps):
                action = self.agent.act(state, greedy=True)
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
    # Inizializza la run di wandb (leggerà la configurazione fornita dall'agent)
    wandb.init()
    config = wandb.config

    print(
        f"Avvio run con: alpha={config.alpha:.3f}, gamma={config.gamma:.3f}, eps_decay={config.eps_decay:.4f}"
    )

    # Creazione ambiente
    cfg_env = RewardConfig()
    env = make_env(reward_config=cfg_env)
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

    # Addestramento
    # Usiamo meno episodi per lo sweep per velocizzare, es. 3000 invece di 5000
    trainer.train(episodes=3000, log_every=500)

    # Valutazione finale (questa è la metrica che WandB cercherà di massimizzare)
    eval_reward, eval_success = trainer.evaluate(episodes=50)

    # WandB logga già in Trainer.evaluate(), ma lo assicuriamo per l'ottimizzatore Bayesiano
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
    cfg = RewardConfig()
    env = make_env(reward_config=cfg)
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
    wandb.finish()  # Chiude la connessione con WandB

    # ==========================================
    # TEST VISIVO FINALE (ANIMAZIONE)
    # ==========================================
    print("\n" + "=" * 40)
    print("Avvio test visivo dell'agente addestrato!")
    print("=" * 40)

    env_vis = make_env(render_mode="human", reward_config=cfg)
    test_episodes = 10

    for ep in range(test_episodes):
        obs, info = env_vis.reset()
        state = encoder.encode(env_vis, info)
        done = False

        print(f"Episodio visivo {ep + 1}/{test_episodes} in corso...")

        prev_stage = None
        step_num = 0

        while not done and step_num < 300:
            action = agent.act(state, greedy=True)
            obs, reward, terminated, truncated, info = env_vis.step(action)
            state = encoder.encode(env_vis, info)
            done = terminated or truncated
            step_num += 1

            # --- Progresso per stage ---
            curr_stage = info.get("stage", "?")
            completion = info.get("completion", 0.0)  # 0.0–1.0 globale
            curr_progress = env_vis.get_wrapper_attr("curr_progress")
            stage_labels = {
                "no_key": "1/4 - Raccogli la chiave",
                "has_key": "2/4 - Apri la porta",
                "door_open": "3/4 - Raggiungi il goal",
                "goal_reached": "4/4 - Goal raggiunto!  ✓",
            }
            label = stage_labels.get(curr_stage, curr_stage)

            bar_len = 20
            filled = int(curr_progress * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"  step {step_num:3d} | Stage: {label:35s} "
                f"| Progresso stage: [{bar}] {curr_progress*100:5.1f}% "
                f"| Completamento: {completion*100:5.1f}%"
            )
            prev_stage = curr_stage

            time.sleep(0.15)

        print(
            f"  → Episodio terminato in {step_num} step. Stage finale: {info.get('stage', '?')}\n"
        )
        time.sleep(1.0)

    env_vis.close()


if __name__ == "__main__":
    main()

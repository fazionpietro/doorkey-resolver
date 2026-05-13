#!/usr/bin/env python3
import sys
import time
import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import cast

import numpy as np
import wandb
import gymnasium as gym
from gymnasium.spaces import Discrete

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.factory import make_env
from env.rewardsystem import RewardConfig
from env.vlm_wrapper import VLMDebugWrapper


class StateEncoder:
    def encode(self, env, info):
        base = env.unwrapped
        ax, ay = base.agent_pos
        d = base.agent_dir

        stage = env.get_wrapper_attr("curr_stage")
        curr_progress = env.get_wrapper_attr("curr_progress")
        progress_bin = int(curr_progress * 9)

        stage_name = stage.value if stage is not None else "no_key"

        if stage_name == "no_key":
            target_pos = env.get_wrapper_attr("key_pos")
        elif stage_name == "has_key":
            target_pos = env.get_wrapper_attr("door_pos")
        elif stage_name == "door_open":
            target_pos = env.get_wrapper_attr("goal_pos")
        else:
            target_pos = None

        tx, ty = target_pos if target_pos is not None else (ax, ay)
        dx = tx - ax
        dy = ty - ay

        stage_map = {"no_key": 0, "has_key": 1, "door_open": 2, "goal_reached": 3}
        stage_idx = stage_map.get(stage_name, 0)

        return (dx, dy, d, progress_bin, stage_idx)


class QLearningAgent:
    def __init__(
        self,
        n_actions: int = 7,
        alpha: float = 0.15,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def act(self, state) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q[state]
        max_q = np.max(q_values)
        best = np.flatnonzero(q_values == max_q)
        return int(np.random.choice(best))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.q[s_next])
        td_target = r + self.gamma * best_next
        self.q[s][a] += self.alpha * (td_target - self.q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Trainer:
    def __init__(self, env, agent: QLearningAgent, encoder: StateEncoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder

    def train(self, episodes: int = 5000, max_steps: int = 100, log_every: int = 100):
        rewards_history = []
        success_buffer = deque(maxlen=100)

        stage_map = {
            "no_key": 0,
            "has_key": 1,
            "door_open": 2,
            "goal_reached": 3,
            "unknown": -1,
        }
        success_rate = 0.0

        for ep in range(episodes):

            obs, info = self.env.reset()

            state = self.encoder.encode(self.env, info)
            ep_reward = 0.0
            info_next = info

            for step in range(max_steps):
                action = self.agent.act(state)
                if ep <= 150:
                    obs_next, reward, terminated, truncated, info_next = self.env.step(
                        action, ignore=True
                    )
                else:
                    self.agent.decay_epsilon()
                    obs_next, reward, terminated, truncated, info_next = self.env.step(
                        action, ignore=False
                    )
                done = terminated or truncated

                next_state = self.encoder.encode(self.env, info_next)
                self.agent.update(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward

                if done:
                    break

            rewards_history.append(ep_reward)
            final_stage = info_next.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value

            is_success = int(final_stage == "goal_reached")
            success_buffer.append(is_success)
            success_rate = float(np.mean(success_buffer))
            vlm_progress = getattr(self.env, "last_progress", 0.0)

            wandb.log(
                {
                    "train/episode": ep,
                    "train/reward": ep_reward,
                    "train/steps": step + 1,
                    "train/epsilon": self.agent.epsilon,
                    "train/success": is_success,
                    "train/success_rate_100ep": success_rate,
                    "train/final_stage_idx": stage_map.get(final_stage, -1),
                    "train/vlm_progress": vlm_progress,
                }
            )

            if ep % log_every == 0:
                avg = float(np.mean(rewards_history[-log_every:]))
                final_stage = info_next.get("stage", "unknown")
                if hasattr(final_stage, "value"):
                    final_stage = final_stage.value
                vlm_progress = getattr(self.env, "last_progress", 0.0)

                print(
                    f"Ep {ep:5d} | reward={ep_reward:7.2f} | avg={avg:7.2f} | succ={success_rate:.2f} | ε={self.agent.epsilon:.3f} | stage={final_stage} | prog={vlm_progress:.1f}%"
                )

        return rewards_history

    def evaluate(self, episodes: int = 100, max_steps: int = 300):
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

        avg_reward = float(np.mean(rewards))
        success_rate = successes / episodes

        wandb.log({"eval/avg_reward": avg_reward, "eval/success_rate": success_rate})
        return avg_reward, success_rate


class StripRewardWrapper(gym.RewardWrapper):
    """Azzera QUALSIASI reward calcolata dai wrapper sottostanti (incluso RewardConfig)."""

    def reward(self, reward):
        return 0.0


def build_env(reward_config: RewardConfig, render_mode: str | None = None) -> any:
    env = make_env(reward_config=reward_config, render_mode=render_mode, size=5)
    env = StripRewardWrapper(
        env
    )  # <-- Blocca totalmente il reward di base/RewardConfig
    env = VLMDebugWrapper(env, query_every=10)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--project_name", type=str, default="doorkey-qlearning")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_eps", type=int, default=100)
    parser.add_argument("--visual_eps", type=int, default=5)
    args = parser.parse_args()

    wandb.init(
        project=args.project_name,
        name=f"vlm_ep{args.episodes}_a{args.alpha}",
        config=vars(args),
    )

    cfg = RewardConfig()
    env = build_env(reward_config=cfg)
    n_actions = int(cast(Discrete, env.action_space).n)

    encoder = StateEncoder()
    agent = QLearningAgent(
        n_actions=n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.eps_decay,
    )
    trainer = Trainer(env, agent, encoder)

    trainer.train(episodes=args.episodes, log_every=args.log_every)

    agent.epsilon = agent.epsilon_min
    avg_r, succ = trainer.evaluate(episodes=args.eval_eps)
    print(f"Eval: avg_reward={avg_r:.2f} | success_rate={succ*100:.1f}%")

    env.close()
    wandb.finish()

    if args.visual_eps > 0:
        env_vis = build_env(reward_config=cfg, render_mode="human")
        for ep in range(args.visual_eps):
            obs, info = env_vis.reset()
            state = encoder.encode(env_vis, info)
            done = False
            step_num = 0

            while not done and step_num < 300:
                action = agent.act(state)
                obs, reward, terminated, truncated, info = env_vis.step(action)
                state = encoder.encode(env_vis, info)
                done = terminated or truncated
                step_num += 1
                time.sleep(0.15)
        env_vis.close()


if __name__ == "__main__":
    main()

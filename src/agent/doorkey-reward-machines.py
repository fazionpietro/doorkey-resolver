#!/usr/bin/env python3
import sys
from pathlib import Path
from random import seed
import time
from collections import defaultdict, deque
import argparse
import numpy as np
import gymnasium as gym
from typing import cast
from gymnasium.spaces import Discrete


sys.path.insert(0, str(Path(__file__).parent.parent))
from env.factory import make_env
from env.rewardsystem import RewardConfig

SEED = 42


class StateEncoder:
    def encode(self, env, info):
        base = env.unwrapped
        ax, ay = base.agent_pos
        d = base.agent_dir

        stage = env.get_wrapper_attr("curr_stage")
        curr_progress = env.curr_progress
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
        dx, dy = tx - ax, ty - ay

        stage_map = {"no_key": 0, "has_key": 1, "door_open": 2, "goal_reached": 3}
        stage_idx = stage_map.get(stage_name, 0)

        return (dx, dy, d, progress_bin, stage_idx)


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
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.q[s_next])
        td_target = r + self.gamma * best_next
        self.q[s][a] += self.alpha * (td_target - self.q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Trainer:
    def __init__(self, env, agent, encoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder

    def train(self, episodes=5000, max_steps=300, log_every=100):
        rewards = []
        success_buffer = deque(maxlen=100)

        for ep in range(episodes):

            if ep == 0:
                obs, info = self.env.reset(seed=SEED)
            else:
                obs, info = self.env.reset()
            state = self.encoder.encode(self.env, info)
            ep_reward = 0.0
            info_last = info

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
                info_last = info_next
                if done:
                    break

            self.agent.decay_epsilon()
            rewards.append(ep_reward)

            final_stage = info_last.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value

            is_success = 1 if final_stage == "goal_reached" else 0
            success_buffer.append(is_success)

            if ep % log_every == 0:
                avg = np.mean(rewards[-log_every:])
                curr_succ = np.mean(success_buffer)
                print(
                    f"Ep {ep:5d} | Reward: {avg:6.2f} | Success: {curr_succ*100:5.1f}% | Eps: {self.agent.epsilon:.3f}"
                )

        return rewards

    def evaluate(self, episodes=50, max_steps=300):
        total_rewards = []
        successes = 0
        for _ in range(episodes):
            obs, info = self.env.reset()
            state = self.encoder.encode(self.env, info)
            ep_reward = 0.0
            done = False
            while not done:
                action = self.agent.act(state, greedy=True)
                obs, reward, term, trunc, info = self.env.step(action)
                state = self.encoder.encode(self.env, info)
                ep_reward += reward
                done = term or trunc

            total_rewards.append(ep_reward)
            if info.get("stage") == "goal_reached" or (
                hasattr(info.get("stage"), "value")
                and info.get("stage").value == "goal_reached"
            ):
                successes += 1

        return np.mean(total_rewards), successes / episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    args = parser.parse_args()

    np.random.seed(SEED)  # Seed globale per riproducibilità

    cfg = RewardConfig()
    env = make_env(reward_config=cfg)
    n_actions = int(cast(Discrete, env.action_space).n)

    agent = QLearningAgent(
        n_actions=n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.eps_decay,
    )
    trainer = Trainer(env, agent, StateEncoder())

    print(f"--- Inizio Training ({args.episodes} episodi) ---")
    trainer.train(episodes=args.episodes)

    print("\n--- Valutazione Finale ---")
    avg_r, succ_r = trainer.evaluate()
    print(f"Reward medio: {avg_r:.2f} | Success Rate: {succ_r*100:.1f}%")
    env.close()

    # Test visivo
    print("\nAvvio test visivo (3 episodi)...")
    env_vis = make_env(render_mode="human", reward_config=cfg)
    for ep in range(3):
        obs, info = env_vis.reset()
        done = False
        while not done:
            state = StateEncoder().encode(env_vis, info)
            action = agent.act(state, greedy=True)
            obs, reward, term, trunc, info = env_vis.step(action)
            done = term or trunc
            time.sleep(0.1)
    env_vis.close()


if __name__ == "__main__":
    main()

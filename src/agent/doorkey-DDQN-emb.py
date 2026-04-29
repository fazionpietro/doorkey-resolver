#!/usr/bin/env python3
import sys
import argparse
import random
import time
from pathlib import Path
from typing import cast
from collections import deque

import wandb
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Assicurati che questi percorsi siano corretti nel tuo progetto
sys.path.insert(0, str(Path(__file__).parent.parent))
from env.factory import make_env
from env.rewardsystem import RewardConfig, Stage, DoorKeyRewardSystem
from ExperienceReplayBuffer import ExperienceReplayBuffer, Experience

# ─────────────────────────────────────────────
# Costante hardcodata per il progetto
# ─────────────────────────────────────────────
PROJECT_NAME = "doorkey-qlearning"
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Reti Neurali
# ─────────────────────────────────────────────
class MiniGridEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.object_emb = nn.Embedding(num_embeddings=20, embedding_dim=4)
        self.state_emb = nn.Embedding(num_embeddings=10, embedding_dim=2)

    def forward(self, image):
        image = image.long()
        obj_ids = image[:, :, :, 0]
        state_ids = image[:, :, :, 2]

        obj_features = self.object_emb(obj_ids)
        state_features = self.state_emb(state_ids)

        # Output: 4 + 2 = 6 canali totali
        combined = torch.cat([obj_features, state_features], dim=-1)
        return combined.permute(0, 3, 1, 2)


class DualHeadDDQN(nn.Module):
    def __init__(self, action_dim=5, env_size=5):
        super().__init__()

        self.embedding = MiniGridEmbedding()

        self.flatten = nn.Flatten()

        # 3. Rete MLP per la griglia (6 canali * 8 larghezza * 8 altezza = 384)
        input_size = 6 * env_size * env_size
        self.grid_mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.scalar_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
        )

        # 5. Corpo finale che unisce i due flussi
        self.fc = nn.Sequential(
            nn.Linear(144, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward(self, image, features):
        # Passa dagli embedding e poi appiattisci tutto
        emb_out = self.embedding(image)
        flat_grid = self.flatten(emb_out)

        # Elabora i due flussi
        grid_out = self.grid_mlp(flat_grid)
        scalar_out = self.scalar_mlp(features)

        # Unisci e calcola i Q-Values
        combined = torch.cat((grid_out, scalar_out), dim=1)
        return self.fc(combined)


# ─────────────────────────────────────────────
# Wrapper Osservazioni
# ─────────────────────────────────────────────
class DDQNObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        env = FullyObsWrapper(env)
        super().__init__(env)

        obs_space = cast(Dict, self.env.observation_space)
        img_space = obs_space["image"]

        self.observation_space = Dict(
            {
                "image": img_space,
                "features": Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32),
            }
        )

    def observation(self, observation: dict) -> dict:
        base_env = cast(MiniGridEnv, self.unwrapped)
        curr_wrapper = self.env
        reward_sys = None

        while hasattr(curr_wrapper, "env"):
            if isinstance(curr_wrapper, DoorKeyRewardSystem):
                reward_sys = curr_wrapper
                break
            curr_wrapper = getattr(curr_wrapper, "env")

        if reward_sys is None and isinstance(curr_wrapper, DoorKeyRewardSystem):
            reward_sys = curr_wrapper

        if reward_sys is None:
            raise RuntimeError(
                "DoorKeyRewardSystem non trovato nello stack dell'ambiente."
            )

        stage = reward_sys.curr_stage
        progress = reward_sys.curr_progress

        stage_idx = 0.0
        if stage == Stage.HAS_KEY:
            stage_idx = 1.0
        elif stage == Stage.DOOR_OPEN:
            stage_idx = 2.0
        elif stage == Stage.GOAL_REACHED:
            stage_idx = 3.0

        if stage == Stage.NO_KEY or stage is None:
            target = reward_sys.key_pos
        elif stage == Stage.HAS_KEY:
            target = reward_sys.door_pos
        else:
            target = reward_sys.goal_pos

        if target is None:
            target = (0, 0)

        agent_pos = base_env.agent_pos
        agent_dir = base_env.agent_dir

        dx = target[0] - agent_pos[0]
        dy = target[1] - agent_pos[1]

        # features = np.array([dx, dy, agent_dir, progress, stage_idx], dtype=np.float32)
        features = np.array([progress, (stage_idx / 3.0)], dtype=np.float32)

        return {
            "image": observation["image"],
            "features": features,
        }


# ─────────────────────────────────────────────
# Agente PER DDQN
# ─────────────────────────────────────────────
class PERDDQNAgent:
    def __init__(
        self,
        action_dim=7,
        lr=1e-4,
        gamma=0.99,
        eps_decay=0.99997,
        buffer_size=100000,
        batch_size=64,
        device: torch.device = torch.device("cpu"),
        env_size=7,
    ):
        self.device = device
        self.action_dim = action_dim

        self.policy_net = DualHeadDDQN(action_dim, env_size).to(self.device)
        self.target_net = DualHeadDDQN(action_dim, env_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=5e-4)

        self.memory = ExperienceReplayBuffer(
            batch_size=batch_size,
            buffer_size=buffer_size,
            alpha=0.4,
            random_state=np.random.RandomState(),
        )

        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay: float = eps_decay
        self.tau = 0.005

        self.beta = 0.5
        self.beta_increment = 0.00001

    def select_action(self, state, evaluate=False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        img_t = torch.tensor(state["image"]).unsqueeze(0).to(self.device)
        feat_t = torch.tensor(state["features"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(img_t, feat_t)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.memory) < self.memory.batch_size:
            return 0.0

        idxs, experiences, weights = self.memory.sample(beta=self.beta)

        states = [e.state for e in experiences]
        next_states = [e.next_state for e in experiences]

        img = np.array([s["image"] for s in states])
        feat = np.array([s["features"] for s in states])
        next_img = np.array([s["image"] for s in next_states])
        next_feat = np.array([s["features"] for s in next_states])

        act = np.array([e.action for e in experiences])
        rew = np.array([e.reward for e in experiences], dtype=np.float32)
        done = np.array([e.done for e in experiences], dtype=np.float32)

        img_t = torch.tensor(img).to(self.device)
        feat_t = torch.tensor(feat).to(self.device)
        act_t = torch.tensor(act, dtype=torch.int64).unsqueeze(1).to(self.device)
        rew_t = torch.tensor(rew).unsqueeze(1).to(self.device)
        next_img_t = torch.tensor(next_img).to(self.device)
        next_feat_t = torch.tensor(next_feat).to(self.device)
        done_t = torch.tensor(done).unsqueeze(1).to(self.device)

        weights_t = (
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

        current_q = self.policy_net(img_t, feat_t).gather(1, act_t)

        with torch.no_grad():
            best_next_actions = self.policy_net(next_img_t, next_feat_t).argmax(
                dim=1, keepdim=True
            )
            next_q = self.target_net(next_img_t, next_feat_t).gather(
                1, best_next_actions
            )
            target_q = rew_t + (self.gamma * next_q * (1 - done_t))

        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        priorities = td_errors.flatten() + 1e-5
        self.memory.update_priorities(idxs, priorities)

        max_td_error = float(td_errors.max())
        wandb.log(
            {
                "debug/max_td_error": max_td_error,
                "debug/mean_q_value": current_q.mean().item(),
            },
            commit=False,
        )

        elementwise_loss = nn.functional.smooth_l1_loss(
            current_q, target_q, reduction="none"
        )
        loss = (weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return loss.item()

    def update_target_network(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_target_network2(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_epsilon2(self):
        delta: float = (1 - self.epsilon_min) / self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon - delta)
        return


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class TrainerDDQN:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, episodes=3000, max_steps=400, log_every=100):
        rewards = []
        success_buffer = deque(maxlen=100)
        count = 0

        for ep in range(episodes):
            state, info = self.env.reset()
            ep_reward = 0.0

            ep_loss = 0.0
            steps_taken = 0
            loss = 0.0

            for step in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done_for_q = terminated
                episode_ended = terminated or truncated

                exp = Experience(state, action, reward, next_state, done_for_q)
                self.agent.memory.add(exp)

                if count % 4 == 0:
                    loss = self.agent.update()
                    # self.agent.update_target_network()

                if count % 9000 == 0:
                    self.agent.update_target_network2()

                ep_loss += float(loss) if loss is not None else 0.0
                ep_reward += float(reward)

                state = next_state
                steps_taken += 1

                # if ep > 1000:
                #    self.agent.decay_epsilon()

                self.agent.decay_epsilon2()

                count += 1
                if episode_ended:
                    break

            rewards.append(ep_reward)
            final_stage = info.get("stage", "unknown")
            is_success = 1 if final_stage == "goal_reached" else 0
            success_buffer.append(is_success)

            current_success_rate = np.mean(success_buffer)

            avg_loss = ep_loss / steps_taken if steps_taken > 0 else 0.0

            avg_r = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)

            wandb.log(
                {
                    "train/episode": ep,
                    "train/reward": ep_reward,
                    "train/steps": steps_taken,
                    "train/epsilon": self.agent.epsilon,
                    "train/success": is_success,
                    "train/success_rate_100ep": current_success_rate,
                    "train/avg_loss": avg_loss,
                    "train/loss": ep_loss,
                },
            )

            if ep % log_every == 0:

                print(
                    f"Ep {ep:5d}: reward={ep_reward:.2f}, avg_100={avg_r:.2f}, "
                    f"succ_rate={current_success_rate:.2f}, eps={self.agent.epsilon:.3f}, loss={avg_loss:.4f}"
                )

            if self.agent.epsilon == self.agent.epsilon_min and count % 500 == 0:
                _, success_rate = self.evaluate()
                if success_rate > 0.90:
                    break

    def evaluate(self, episodes=50, max_steps=300):
        rewards = []
        successes = 0
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        for _ in range(episodes):
            state, info = self.env.reset()
            ep_reward = 0.0
            for _ in range(max_steps):
                action = self.agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            final_stage = info.get("stage", "unknown")
            if final_stage == 3 or final_stage == "goal_reached":
                successes += 1

        self.agent.epsilon = original_epsilon
        avg_reward = float(np.mean(rewards))
        success_rate = successes / episodes

        wandb.log({"eval/avg_reward": avg_reward, "eval/success_rate": success_rate})
        return avg_reward, success_rate


# ─────────────────────────────────────────────
# Main & Sweep
# ─────────────────────────────────────────────
def train_sweep_ddqn():
    wandb.init(project=PROJECT_NAME)
    config = wandb.config
    set_seed(SEED)

    cfg_env = RewardConfig()
    env = make_env(reward_config=cfg_env)
    env = DDQNObservationWrapper(env)

    n_actions = int(cast(Discrete, env.action_space).n)

    obs_space = cast(Dict, env.observation_space)
    obs_shape = obs_space["image"].shape
    env_width = obs_shape[0]

    agent = PERDDQNAgent(
        action_dim=n_actions,
        lr=config.lr,
        gamma=config.gamma,
        eps_decay=config.eps_decay,
        buffer_size=1_000_000,
        batch_size=256,
        device=device,
        env_size=env_width,
    )

    trainer = TrainerDDQN(env, agent)
    trainer.train(episodes=3000, max_steps=400, log_every=200)

    eval_reward, eval_success = trainer.evaluate(episodes=50)
    wandb.log({"sweep/final_success_rate": eval_success})
    env.close()


def main():
    parser = argparse.ArgumentParser(description="DoorKey DDQN Vision con WandB.")
    parser.add_argument("--mode", type=str, choices=["train", "sweep"], default="train")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.99995)
    parser.add_argument("--sweep_count", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "sweep":
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "eval/success_rate", "goal": "maximize"},
            "parameters": {
                "lr": {"min": 5e-5, "max": 1e-3},
                "gamma": {"min": 0.99, "max": 0.995},
                "eps_decay": {"min": 0.99998, "max": 0.999999},
            },
        }
        sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
        wandb.agent(sweep_id, function=train_sweep_ddqn, count=args.sweep_count)
        return

    print("Creazione ambiente DoorKey...")
    set_seed(SEED)
    cfg_env = RewardConfig()
    env = make_env(reward_config=cfg_env)
    env = DDQNObservationWrapper(env)

    n_actions = int(cast(Discrete, env.action_space).n)

    obs_space = cast(Dict, env.observation_space)
    obs_shape = obs_space["image"].shape
    env_width = obs_shape[0]

    wandb.init(
        project=PROJECT_NAME,
        name=f"DDQN_ep{args.episodes}_lr{args.lr}",
        config={
            "episodes": args.episodes,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "epsilon_decay": args.eps_decay,
            "env_id": "MiniGrid-DoorKey",
            "agent_type": "DDQN_DualHead_PER",
        },
    )

    agent = PERDDQNAgent(
        action_dim=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        eps_decay=args.eps_decay,
        buffer_size=1_000_000,
        batch_size=256,
        device=device,
        env_size=env_width,
    )

    trainer = TrainerDDQN(env=env, agent=agent)
    print("Training avviato...")
    trainer.train(episodes=args.episodes, log_every=100)

    trainer.evaluate()
    env.close()
    wandb.finish()

    # TEST VISIVO (Rimosso per brevità, ma usa la stessa logica di env_width)


if __name__ == "__main__":
    main()

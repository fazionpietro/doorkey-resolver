#!/usr/bin/env python3
import sys
from pathlib import Path
import time
from collections import deque
import argparse
import random
import numpy as np
import gymnasium as gym
from typing import cast
from gymnasium.spaces import Discrete
import wandb

# Import di PyTorch per il Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import dei Wrapper di MiniGrid per la visione
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# Mock imports - Assicurati che questi percorsi siano corretti nel tuo progetto
sys.path.insert(0, str(Path(__file__).parent.parent))
from env.factory import make_env
from env.rewardsystem import RewardConfig


# ─────────────────────────────────────────────
# QNetwork: Rete Neurale standard (MLP)
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


# ─────────────────────────────────────────────
# DDQNAgent: Agente Double Deep Q-Network
# ─────────────────────────────────────────────
class DDQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """Sceglie un'azione usando una policy epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state, done):
        """Addestra la rete su una singola transizione (Niente Replay Buffer)"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_t = torch.tensor([action]).to(self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done_t = torch.tensor([float(done)], dtype=torch.float32).to(self.device)

        # FASE 1: Calcolo del Q-value corrente
        q_values = self.online_net(state_t)
        current_q = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # FASE 2: Calcolo del Target DDQN
        with torch.no_grad():
            next_action = self.online_net(next_state_t).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_t)
            next_q = next_q_values.gather(1, next_action).squeeze(1)
            target_q = reward_t + (1 - done_t) * self.gamma * next_q

        # FASE 3: Ottimizzazione
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Aggiorna i pesi della Target Network copiando quelli della Online Network"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        """Riduce l'esplorazione col tempo"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────────
# Encoder Visivo
# ─────────────────────────────────────────────
class FullVisionEncoder:
    def __init__(self):
        pass

    def encode(self, obs):
        """Appiattisce la matrice 3D e normalizza i valori dividendo per 10.0"""
        flat_obs = obs.flatten() / 10.0
        return flat_obs


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class TrainerDDQN:
    def __init__(self, env, agent, encoder, update_target_every=500):
        self.env = env
        self.agent = agent
        self.encoder = encoder
        self.update_target_every = update_target_every
        self.total_steps = 0

    def train(self, episodes=10000, max_steps=300, log_every=100):
        rewards = []
        avg_rewards = []
        success_buffer = deque(maxlen=100)

        for ep in range(episodes):
            obs, info = self.env.reset()
            state = self.encoder.encode(obs)
            reward = 0

            ep_reward = 0.0
            ep_loss = 0.0
            steps_taken = 0
            info_next = info

            for step in range(max_steps):
                action = self.agent.act(state)

                obs_next, reward, terminated, truncated, info_next = self.env.step(
                    action
                )
                done = terminated or truncated
                next_state = self.encoder.encode(obs_next)

                loss = self.agent.train(state, action, reward, next_state, done)
                ep_loss += loss

                self.total_steps += 1
                if self.total_steps % self.update_target_every == 0:
                    self.agent.update_target_network()

                state = next_state
                ep_reward += reward
                steps_taken += 1

                if done:
                    break

            self.agent.decay_epsilon()
            rewards.append(ep_reward)

            final_stage = info_next.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value

            is_success = 1 if final_stage == "goal_reached" or reward > 0 else 0
            success_buffer.append(is_success)
            current_success_rate = (
                np.mean(success_buffer) if len(success_buffer) > 0 else 0
            )

            wandb.log(
                {
                    "train/episode": ep,
                    "train/reward": ep_reward,
                    "train/steps": steps_taken,
                    "train/epsilon": self.agent.epsilon,
                    "train/success": is_success,
                    "train/success_rate_100ep": current_success_rate,
                    "train/avg_loss": ep_loss / steps_taken if steps_taken > 0 else 0,
                }
            )

            if ep % log_every == 0:
                avg = (
                    np.mean(rewards[-log_every:])
                    if len(rewards) >= log_every
                    else np.mean(rewards)
                )
                avg_rewards.append(avg)
                print(
                    f"Ep {ep:5d}: reward={ep_reward:.2f}, avg={avg:.2f}, "
                    f"succ_rate={current_success_rate:.2f}, ε={self.agent.epsilon:.3f}, loss_media={ep_loss/steps_taken:.4f}"
                )

        return rewards, avg_rewards

    def evaluate(self, episodes=50, max_steps=300):
        """Valuta l'agente spegnendo l'esplorazione epsilon-greedy"""
        rewards = []
        successes = 0

        # Salviamo l'epsilon corrente e forziamo la policy ad essere avida (greedy)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        for _ in range(episodes):
            obs, info = self.env.reset()
            state = self.encoder.encode(obs)
            ep_reward = 0.0
            info_next = info

            for _ in range(max_steps):
                action = self.agent.act(state)
                obs, reward, terminated, truncated, info_next = self.env.step(action)
                state = self.encoder.encode(obs)
                ep_reward += reward
                if terminated or truncated:
                    break

            rewards.append(ep_reward)

            final_stage = info_next.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value
            if final_stage == "goal_reached":
                successes += 1

        # Ripristiniamo l'epsilon originale
        self.agent.epsilon = original_epsilon

        avg_reward = np.mean(rewards)
        success_rate = successes / episodes

        # Log evaluation metrics
        wandb.log({"eval/avg_reward": avg_reward, "eval/success_rate": success_rate})

        return avg_reward, success_rate


# ─────────────────────────────────────────────
# Funzioni Main e Sweep
# ─────────────────────────────────────────────
def train_sweep_ddqn():
    """Funzione chiamata automaticamente dai WandB Agents durante lo sweep"""
    wandb.init()
    config = wandb.config

    print(
        f"Avvio run sweep: lr={config.lr:.5f}, gamma={config.gamma:.3f}, eps_decay={config.eps_decay:.4f}"
    )

    cfg_env = RewardConfig()
    env = make_env(reward_config=cfg_env)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    obs_shape = env.observation_space.shape
    assert obs_shape is not None, "L'observation space non ha una shape!"
    state_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
    n_actions = int(cast(Discrete, env.action_space).n)

    encoder = FullVisionEncoder()
    agent = DDQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=config.lr,
        gamma=config.gamma,
        epsilon_decay=config.eps_decay,
    )

    trainer = TrainerDDQN(env, agent, encoder, update_target_every=500)
    trainer.train(episodes=3000, max_steps=300, log_every=500)

    # Valutazione a fine sweep
    eval_reward, eval_success = trainer.evaluate(episodes=50)
    wandb.log({"sweep/final_success_rate": eval_success})
    print(f"Run conclusa. Success Rate (Eval): {eval_success*100:.1f}%")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="DoorKey DDQN Vision con WandB.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "sweep"],
        default="train",
        help="'train' per singola run, 'sweep' per WandB sweep agent",
    )
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--project_name", type=str, default="doorkey-ddqn-vision")
    parser.add_argument(
        "--sweep_count", type=int, default=10, help="Numero di run nello sweep"
    )

    args = parser.parse_args()

    if args.mode == "sweep":
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "eval/success_rate", "goal": "maximize"},
            "parameters": {
                "lr": {"min": 1e-4, "max": 1e-2},
                "gamma": {"min": 0.90, "max": 0.999},
                "eps_decay": {"min": 0.995, "max": 0.9995},
            },
        }
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        print(f"Sweep creato: id='{sweep_id}' | count={args.sweep_count}")
        wandb.agent(sweep_id, function=train_sweep_ddqn, count=args.sweep_count)
        return

    print("Creazione ambiente DoorKey 6x6...")
    cfg = RewardConfig()
    env = make_env(reward_config=cfg)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    obs_shape = env.observation_space.shape
    assert obs_shape is not None, "L'observation space non ha una shape definita!"

    state_dim = obs_shape[0] * obs_shape[1] * obs_shape[2]
    n_actions = int(cast(Discrete, env.action_space).n)

    print(f"Dimensione stato (Pixel appiattiti): {state_dim}")
    print(f"Azioni disponibili: {n_actions}")

    wandb.init(
        project=args.project_name,
        name=f"DDQN_ep{args.episodes}_lr{args.lr}",
        config={
            "episodes": args.episodes,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "epsilon_decay": args.eps_decay,
            "env_id": "MiniGrid-DoorKey-6x6-v0",
            "agent_type": "DDQN_FullVision_NoBuffer",
        },
    )

    encoder = FullVisionEncoder()
    agent = DDQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.eps_decay,
    )

    trainer = TrainerDDQN(
        env=env, agent=agent, encoder=encoder, update_target_every=500
    )

    print("Training avviato...")
    trainer.train(episodes=args.episodes, log_every=100)

    print("Valutazione...")
    eval_reward, eval_success = trainer.evaluate()
    print(f"Evaluation reward medio: {eval_reward:.2f}")
    print(f"Evaluation success rate: {eval_success*100:.1f}%")

    env.close()
    wandb.finish()

    # ========================================================
    # TEST VISIVO FINALE
    # ========================================================
    print("\n" + "=" * 40)
    print("Avvio test visivo dell'agente addestrato!")
    print("=" * 40)

    # Dobbiamo applicare i wrapper visivi anche qui per non far crashare l'encoder
    env_vis = make_env(render_mode="human", reward_config=cfg)
    env_vis = FullyObsWrapper(env_vis)
    env_vis = ImgObsWrapper(env_vis)

    test_episodes = 10

    # Rendiamo l'agente 100% greedy per il test visivo
    agent.epsilon = 0.0

    for ep in range(test_episodes):
        obs, info = env_vis.reset()
        state = encoder.encode(obs)
        done = False
        rewards = 0.0

        print(f"Episodio visivo {ep + 1}/{test_episodes} in corso...")

        step_num = 0

        while not done and step_num < 300:
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env_vis.step(action)
            state = encoder.encode(obs)
            done = terminated or truncated
            step_num += 1
            rewards += float(reward)

            # --- Progresso per stage ---
            curr_stage = info.get("stage", "?")
            if hasattr(curr_stage, "value"):
                curr_stage = curr_stage.value

            completion = info.get("completion", 0.0)
            curr_progress = env_vis.get_wrapper_attr("curr_progress")
            stage_labels = {
                "no_key": "1/4 - Raccogli la chiave",
                "has_key": "2/4 - Apri la porta",
                "door_open": "3/4 - Raggiungi il goal",
                "goal_reached": "4/4 - Goal raggiunto!  ✓",
            }
            label = stage_labels.get(curr_stage, str(curr_stage))

            bar_len = 20
            filled = int(curr_progress * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"  step {step_num:3d} | reward: {rewards:3f} | Stage: {label:35s} "
                f"| Progresso stage: [{bar}] {curr_progress*100:5.1f}% "
                f"| Completamento: {completion*100:5.1f}%"
            )

            time.sleep(0.15)

        print(
            f"  → Episodio terminato in {step_num} step. Stage finale: {info.get('stage', '?')}\n"
        )
        time.sleep(1.0)

    env_vis.close()


if __name__ == "__main__":
    main()

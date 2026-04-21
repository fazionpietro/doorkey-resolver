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


# Import di PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import dei Wrapper di MiniGrid
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

# Mock imports - Assicurati che questi percorsi siano corretti nel tuo progetto
sys.path.insert(0, str(Path(__file__).parent.parent))
from env.factory import make_env
from env.rewardsystem import RewardConfig

# Importa il buffer (Assicurati di avere il file ExperienceReplayBuffer.py nella stessa cartella)
from ExperienceReplayBuffer import ExperienceReplayBuffer, Experience

# Usa la GPU se disponibile, altrimenti CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Modello (Policy) che mappa gli stati alle azioni."""
    def __init__(self, state_size, action_size, seed, fc1_units=64, hidden_units=128, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.h1 = nn.Linear(fc1_units, hidden_units)
        self.h2 = nn.Linear(hidden_units, fc2_units)
        self.fc2 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.fc2(x)

class DDQNAgent():
    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, 
                 gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, alpha=0.6,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.995):
        """Inizializza un oggetto DDQNAgent."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        
        # Gestione Epsilon
        self.epsilon = eps_start
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        
        # Q-Network (Local e Target)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Inizializza il Prioritized Experience Replay Buffer
        random_state = np.random.RandomState(seed)
        self.memory = ExperienceReplayBuffer(batch_size, buffer_size, alpha, random_state)
        
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, beta):
        # 1. Salva l'esperienza nel buffer
        exp = Experience(state, action, reward, next_state, done)
        self.memory.add(exp)
        
        loss = 0.0
        # 2. Impara ogni UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(beta)
                loss = self.learn(experiences)
        return loss

    def act(self, state, eps=None):
        """Restituisce l'azione dato lo stato corrente seguendo una policy epsilon-greedy."""
        eps = eps if eps is not None else self.epsilon
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Aggiorna i parametri della rete usando un batch di esperienze."""
        idxs, exps, is_weights = experiences
        
        states = torch.from_numpy(np.vstack([e.state for e in exps if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exps if e is not None]).astype(np.uint8)).float().to(device)
        is_weights = torch.from_numpy(is_weights).float().to(device)

        # LOGICA DOUBLE DQN
        best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # PRIORITIZED EXPERIENCE REPLAY
        td_errors = torch.abs(Q_targets - Q_expected).detach().cpu().numpy()
        new_priorities = td_errors.flatten() + 1e-5 
        self.memory.update_priorities(idxs, new_priorities)

        loss = (is_weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # AGGIORNAMENTO RETE TARGET
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def decay_epsilon(self):
        """Riduce il valore di epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────────
# Encoder Visivo
# ─────────────────────────────────────────────
class FullVisionEncoder:
    def __init__(self):
        pass

    def encode(self, obs):
        """Appiattisce la matrice 3D e normalizza i valori dividendo per 10.0"""
        return obs.flatten() / 10.0


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class TrainerDDQN:
    def __init__(self, env, agent, encoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder
        self.total_steps = 0

    def train(self, episodes=10000, max_steps=300, log_every=100):
        rewards = []
        avg_rewards = []
        success_buffer = deque(maxlen=100)
        
        # Variabili per calcolare il beta (aumenta da 0.4 a 1.0 nel tempo)
        beta_start = 0.4
        reward = 0.0

        for ep in range(episodes):
            obs, info = self.env.reset()
            state = self.encoder.encode(obs)
            
            # Calcolo di Beta per il PER in questo episodio
            beta = min(1.0, beta_start + ep * (1.0 - beta_start) / episodes)

            ep_reward = 0.0
            ep_loss = 0.0
            steps_taken = 0
            info_next = info

            for step in range(max_steps):
                action = self.agent.act(state)
                obs_next, reward, terminated, truncated, info_next = self.env.step(action)
                done = terminated or truncated
                next_state = self.encoder.encode(obs_next)

                # Step dell'agente: salva nel buffer e impara
                loss = self.agent.step(state, action, reward, next_state, done, beta)
                ep_loss += loss

                self.total_steps += 1
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
            current_success_rate = np.mean(success_buffer) if len(success_buffer) > 0 else 0

            wandb.log({
                "train/episode": ep,
                "train/reward": ep_reward,
                "train/steps": steps_taken,
                "train/epsilon": self.agent.epsilon,
                "train/success": is_success,
                "train/success_rate_100ep": current_success_rate,
                "train/avg_loss": ep_loss / steps_taken if steps_taken > 0 else 0,
            })

            if ep % log_every == 0:
                avg = np.mean(rewards[-log_every:]) if len(rewards) >= log_every else np.mean(rewards)
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

        self.agent.epsilon = original_epsilon

        avg_reward = np.mean(rewards)
        success_rate = successes / episodes

        wandb.log({"eval/avg_reward": avg_reward, "eval/success_rate": success_rate})
        return avg_reward, success_rate


# ─────────────────────────────────────────────
# Funzioni Main e Sweep
# ─────────────────────────────────────────────
def train_sweep_ddqn():
    wandb.init()
    config = wandb.config

    print(f"Avvio run sweep: lr={config.lr:.5f}, gamma={config.gamma:.3f}, eps_decay={config.eps_decay:.4f}")

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
        state_size=state_dim,
        action_size=n_actions,
        seed=42,
        buffer_size=100000,
        batch_size=64,
        lr=config.lr,
        gamma=config.gamma,
        eps_decay=config.eps_decay,
    )

    trainer = TrainerDDQN(env, agent, encoder)
    trainer.train(episodes=3000, max_steps=300, log_every=500)

    eval_reward, eval_success = trainer.evaluate(episodes=50)
    wandb.log({"sweep/final_success_rate": eval_success})
    print(f"Run conclusa. Success Rate (Eval): {eval_success*100:.1f}%")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="DoorKey DDQN Vision con WandB.")
    parser.add_argument("--mode", type=str, choices=["train", "sweep"], default="train")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_decay", type=float, default=0.995)
    parser.add_argument("--project_name", type=str, default="doorkey-ddqn-vision")
    parser.add_argument("--sweep_count", type=int, default=10)

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
            "agent_type": "DDQN_FullVision_PER",
        },
    )

    encoder = FullVisionEncoder()
    agent = DDQNAgent(
        state_size=state_dim,
        action_size=n_actions,
        seed=42,
        buffer_size=100000,
        batch_size=64,
        lr=args.lr,
        gamma=args.gamma,
        eps_decay=args.eps_decay,
    )

    trainer = TrainerDDQN(env=env, agent=agent, encoder=encoder)

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

    env_vis = make_env(render_mode="human", reward_config=cfg)
    env_vis = FullyObsWrapper(env_vis)
    env_vis = ImgObsWrapper(env_vis)

    test_episodes = 10
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

            curr_stage = info.get("stage", "?")
            if hasattr(curr_stage, "value"):
                curr_stage = curr_stage.value

            completion = info.get("completion", 0.0)
            
            # Nota: get_wrapper_attr potrebbe richiedere controlli se il wrapper non è standard,
            # ma lo lascio come nel tuo script originario
            try:
                curr_progress = env_vis.get_wrapper_attr("curr_progress")
            except AttributeError:
                curr_progress = 0.0
                
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

        print(f"  → Episodio terminato in {step_num} step. Stage finale: {info.get('stage', '?')}\n")
        time.sleep(1.0)

    env_vis.close()

if __name__ == "__main__":
    main()

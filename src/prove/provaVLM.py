import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from PIL import Image
import gymnasium as gym
import numpy as np
from collections import deque
import re
from minigrid.minigrid_env import MiniGridEnv
from qwen_vl_utils import process_vision_info
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
from minigrid.wrappers import FullyObsWrapper

from env.factory import make_env
from env.rewardsystem import RewardConfig
from env import doorkey_events as doorev

SEED = 42
DEFAULT_ENV_ID = "MiniGrid-DoorKey-5x5-v0"
model_id = "Qwen/Qwen3-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    "Caricamento del modello in corso... (potrebbe richiedere qualche minuto al primo avvio)"
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Forza precisione a 16-bit
    attn_implementation="flash_attention_2",  # Attiva Flash Attention
    device_map="auto",
)
model = torch.compile(model)
processor = AutoProcessor.from_pretrained(model_id)


import re

PROMPT = """You are an expert evaluator for an AI agent. 
This single image contains 5 frames representing the agent's history, concatenated from RIGHT to LEFT:
- The RIGHTMOST part is the oldest state.
- The LEFTMOST part (the 5th frame) is the CURRENT state, which is the direct result of the agent's LAST ACTION.
- the red triangle IS the rappresentation of the agent.

THE RED AGENT'S GOAL IS: reach the green goal using all he need to reach it.

Task:
1. Briefly analyze what changed from the older frames to the newest leftmost frame, understand what there is in the image. 
Understand the shape and the color and understand the enviroment and the puzzle logic.
2. Analize if the last action bring the agent closer to the goal? Was it a good or bad move?
3. After your brief analysis, output exactly 3 metrics inside <METRICS> tags separated by commas.

Metrics to estimate:
- Reward (between -1.0 and 1.0): Immediate quality of the last action.
- Final goal progress (between 0.0 and 1.0): Completion towards the ultimate goal this value is a percentage.
- Next stage progress (between 0.0 and 1.0): Completion towards the immediate next step this value is a percentage.

FORMAT YOUR OUTPUT STRICTLY LIKE THIS dont miss anything for any reason:
<METRICS>reward, final_goal_progress, next_stage_progress</METRICS>
"""
PROMPT = """You are an expert autonomous evaluator for an AI agent in a grid-world puzzle.
This image contains 5 frames representing the agent's history, from RIGHT (oldest) to LEFT (newest/current state).
The red triangle is the agent.

Your task is to DEDUCE the logic of the environment, identify the intermediate goals required to solve it, and evaluate the agent's latest move.

Follow this exact Chain of Thought in your analysis:
1. ENVIRONMENT LOGIC: What objects exist? What is the ultimate goal? What sequence of steps (stages) are logically required? (e.g., Stage 1: Get Key, Stage 2: Open Door, Stage 3: Reach Goal).
2. CURRENT STAGE: Based on the leftmost frame, what is the IMMEDIATE next logical object the agent must interact with or reach?
3. MOVEMENT ANALYSIS: Did the agent's last action in the leftmost frame move it closer to this immediate goal?

After your analysis, output EXACTLY 6 metrics inside a single <METRICS> tag, separated by commas.

Metrics strictly in this order:
1. Stage_ID (int): An arbitrary integer representing the current logical stage (e.g., 1 for finding key, 2 for door, 3 for final goal).
2. Reward (float -1.0 to 1.0): Quality of the very last action towards the current stage goal.
3. Final_Progress (float 0.0 to 1.0): Total percentage of the puzzle completed.
4. Stage_Distance (int): Estimated grid squares between the agent and the CURRENT stage goal.
5. dx (int): Estimated horizontal grid steps to the stage goal (positive = right, negative = left).
6. dy (int): Estimated vertical grid steps to the stage goal (positive = down, negative = up).

STRICT FORMAT EXAMPLE:
<METRICS>1, 0.5, 0.25, 4, 3, -1</METRICS>
"""


def get_vlm_reward(image: Image.Image) -> tuple[float, float, float]:
    """
    Analizza la striscia di immagini, fa ragionare il modello sull'ultima azione
    ed estrae i numeri dai tag <METRICS>.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    # Aumentiamo i token perché ora il modello deve prima scrivere una riga di spiegazione
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = output_text[0].strip()

    metrics_match = re.search(r"<METRICS>(.*?)</METRICS>", response, re.DOTALL)

    return 0.0, 0.0, 0.0


class MoondreamDenseRewardWrapper(gym.Wrapper):
    def __init__(self, env, beta: float = 0.5, query_every: int = 5):
        super().__init__(env)
        self.beta = beta
        self.query_every = query_every
        self._step_count = 0
        self.last_reward = 0.0
        self.last_progress = 0.0
        self.last_stage_progress = 0.0
        self.cache = deque(maxlen=5)

        # Inizializza l'ambiente per ottenere il primo frame
        self.env.reset()
        self._frame_before = self._get_frame()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._frame_before = self._get_frame()
        return obs, info

    def _get_frame(self) -> Image.Image:
        rgb = self.env.unwrapped.render()
        if rgb is None:
            raise ValueError("Inizializza l'env con render_mode='rgb_array'.")
        img = Image.fromarray(np.array(rgb))
        self.cache.append(img)
        return img

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        frame_after = self._get_frame()

        self._step_count += 1

        if self._step_count % self.query_every == 0 and len(self.cache) == 5:
            self.last_reward, self.last_progress, self.last_stage_progress = (
                self._get_reward()
            )

        elif len(self.cache) < 5:
            # Per i primissimi 4 step dell'episodio, il VLM non dà reward
            self.last_reward = 0.0

        total_reward = float(env_reward) + self.beta * self.last_reward
        self._frame_before = frame_after

        info["vlm_reward"] = self.last_reward
        info["vlm_progress"] = self.last_progress
        info["vlm_stage_progress"] = self.last_stage_progress
        info["env_reward"] = env_reward
        info["total_reward"] = total_reward

        return obs, total_reward, terminated, truncated, info

    def _get_reward(self) -> tuple[float, float, float]:

        separator = 10

        width, height = zip(*(img.size for img in self.cache))

        larghezza_totale = sum(width) + (separator * (len(self.cache) - 1))
        altezza_massima = max(height)

        new_img = Image.new("RGB", (larghezza_totale, altezza_massima), "white")

        offset_x = 0
        for i, img in enumerate(self.cache):
            new_img.paste(img, (offset_x, 0))
            offset_x += img.width + separator

        return get_vlm_reward(new_img)


class StateEncoder:

    def encode(self, env, info):
        base = env.unwrapped
        ax, ay = base.agent_pos
        d = base.agent_dir

        return (ax, ay, d)


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
        self.delta = 1 / epsilon_decay

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
        self.epsilon = max(self.epsilon_min, self.epsilon - self.delta)


class Trainer:
    def __init__(self, env, agent, encoder):
        self.env = env
        self.agent = agent
        self.encoder = encoder

    def train(self, episodes=5000, max_steps=300, log_every=100):
        rewards = []
        avg_rewards = []
        success_buffer = deque(maxlen=100)
        current_success_rate = 0.0

        for ep in range(episodes):
            if ep == 0:
                obs, info = self.env.reset(seed=SEED)
            else:
                obs, info = self.env.reset()

            # Lo stato rimane discreto (x, y, dir)
            state = self.encoder.encode(self.env, info)

            ep_reward = 0.0
            ep_vlm_reward = 0.0  # Accumulatore per vedere quanto influisce il VLM
            info_next = info
            steps_taken = 0
            vlm_prog = 0.0
            vlm_stage_prog = 0.0

            for step in range(max_steps):
                action = self.agent.act(state)
                (
                    obs_next,
                    reward,
                    terminated,
                    truncated,
                    info_next,
                ) = self.env.step(action)

                done = terminated or truncated

                # Leggiamo i valori del VLM (per WandB, non per lo stato)
                vlm_r = info_next.get("vlm_reward", 0.0)
                vlm_prog = info_next.get("vlm_progress", 0.0)
                vlm_stage_prog = info_next.get("vlm_stage_progress", 0.0)

                # Lo stato successivo deve avere la stessa forma di quello iniziale
                next_state = self.encoder.encode(self.env, info_next)

                # Aggiornamento standard di Q-Learning
                self.agent.update(state, action, reward, next_state, done)

                self.agent.decay_epsilon()
                state = next_state
                ep_reward += reward
                ep_vlm_reward += vlm_r
                steps_taken += 1

                if done:
                    break

            # Decadimento Epsilon
            rewards.append(ep_reward)

            # Estrazione metriche di fine episodio
            final_stage = info_next.get("stage", "unknown")
            if hasattr(final_stage, "value"):
                final_stage = final_stage.value

            is_success = 1 if final_stage == "goal_reached" else 0
            success_buffer.append(is_success)
            current_success_rate = np.mean(success_buffer)

            # --- LOGGING SU WANDB CORRETTO ---
            wandb.log(
                {
                    "train/episode": ep,
                    "train/reward_totale": ep_reward,
                    "train/reward_vlm_cumulativo": ep_vlm_reward,
                    "train/vlm_progress_finale": vlm_prog,
                    "train/vlm_stage_progress_finale": vlm_stage_prog,
                    "train/epsilon": self.agent.epsilon,
                    "train/success_rate_100ep": current_success_rate,
                    "train/steps": steps_taken,
                }
            )

            # Stampa su console ogni log_every
            if ep % log_every == 0:
                avg = np.mean(rewards[-log_every:])
                avg_rewards.append(avg)
                print(
                    f"Ep {ep:5d}: reward={ep_reward:.2f} (VLM={ep_vlm_reward:.2f}), "
                    f"avg_100={avg:.2f}, succ_rate={current_success_rate:.2f}, "
                    f"ε={self.agent.epsilon:.3f}, stage={final_stage}"
                )

        return rewards, avg_rewards

    def evaluate(self, episodes=100, max_steps=100):
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
    env = FullyObsWrapper(env)

    env = MoondreamDenseRewardWrapper(env)
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
    env = FullyObsWrapper(env)
    env = MoondreamDenseRewardWrapper(env)
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
    rewards, avg_rewards = trainer.train(
        episodes=args.episodes, max_steps=50, log_every=25
    )

    print("Valutazione...")
    eval_reward, eval_success = trainer.evaluate()
    print(f"Evaluation reward medio: {eval_reward:.2f}")
    print(f"Evaluation success rate: {eval_success*100:.1f}%")

    env.close()
    wandb.finish()

    print("\n" + "=" * 40)
    print("Avvio test visivo dell'agente addestrato!")
    print("=" * 40)

    env_vis = make_env(render_mode="human", reward_config=cfg)
    test_episodes = 10

    for ep in range(test_episodes):
        obs, info = env_vis.reset()
        state = encoder.encode(env_vis, info)
        done = False
        rewards = 0

        print(f"Episodio visivo {ep + 1}/{test_episodes} in corso...")

        prev_stage = None
        step_num = 0

        while not done and step_num < 300:
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env_vis.step(action)
            state = encoder.encode(env_vis, info)
            done = terminated or truncated
            step_num += 1
            rewards += reward

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
                f"  step {step_num:3d} | reward: {rewards:3f} | Stage: {label:35s} "
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

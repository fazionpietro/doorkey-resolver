import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import argparse
import numpy as np
import pandas as pd
import ast
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from minigrid.envs import DoorKeyEnv

import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Import del tuo progetto
from env.factory import make_env
from env.rewardsystem import RewardConfig, Stage, DoorKeyRewardSystem

SEED = 42


# ==========================================
# 1. LOGICA DEL CSV & GENERAZIONE DATASET
# ==========================================
def calculate_reward(row):
    x, y = int(row["agent_x"]), int(row["agent_y"])
    d = str(row["agent_dir"]).strip()
    goal_pos = ast.literal_eval(str(row["goal_pos"]).strip())
    tx, ty = goal_pos[0], goal_pos[1]
    goal_name = str(row["goal_name"]).strip()
    action = str(row["action"]).strip()

    if action == "DONE":
        return -0.001
    if action == "DROP":
        return -0.701 if goal_name == "has_key" else -0.001
    if action in ["PICKUP", "TOGGLE"]:
        return -0.001

    if action == "FORWARD":
        nx, ny = x, y
        if d == "west":
            nx = x - 1
        elif d == "est":
            nx = x + 1
        elif d == "north":
            ny = y - 1
        elif d == "sud":
            ny = y + 1

        if nx < 1 or nx > 4 or ny < 1 or ny > 4:
            return -0.001

        dist_before = abs(tx - x) + abs(ty - y)
        dist_after = abs(tx - nx) + abs(ty - ny)

        if dist_after < dist_before:
            return (
                0.08
                if goal_name == "door_open"
                else 0.09 if goal_name == "has_key" else 0.06
            )
        elif dist_after > dist_before:
            return -0.09 if goal_name in ["has_key", "door_open"] else -0.07

    if action in ["LEFT", "RIGHT"]:
        dir_map = {"north": 3, "sud": 1, "est": 0, "west": 2}
        idx = dir_map.get(d, 0)
        nd = (idx - 1) % 4 if action == "LEFT" else (idx + 1) % 4

        def points_to_goal(direction, tx, ty, x, y):
            if direction == 3 and ty < y:
                return True
            if direction == 1 and ty > y:
                return True
            if direction == 0 and tx > x:
                return True
            if direction == 2 and tx < x:
                return True
            return False

        old_red = points_to_goal(idx, tx, ty, x, y)
        new_red = points_to_goal(nd, tx, ty, x, y)

        if new_red and not old_red:
            return 0.026667
        elif old_red and not new_red:
            return -0.049 if goal_name == "has_key" else -0.039667
        else:
            return -0.001
    return -0.001


def generate_dataset(input_csv=None, output_csv=None):
    script_dir = Path(__file__).resolve().parent
    if input_csv is None:
        input_csv = script_dir / "reward_subset_seed1_zero.csv"
    if output_csv is None:
        output_csv = script_dir / "reward_subset_seed1_sero.csv"

    print(f"\n[1] Generazione del dataset da {input_csv.name}...")
    try:
        df = pd.read_csv(input_csv, sep="|", skipinitialspace=True, engine="python")
        df.columns = df.columns.str.strip()
        df["reward"] = df.apply(calculate_reward, axis=1)
        df.to_csv(output_csv, index=False)
        print(f"    Dataset salvato in {output_csv.name} con {len(df)} entry!")
    except Exception as e:
        print(f"    Impossibile generare il dataset CSV: {e}")


# ==========================================
# 2. CONFRONTO LIVE: LA TUA REWARD DENSA vs CSV DENSA
# ==========================================
def confronta_due_reward_dense():
    print("\n[2] CONFRONTO DIRETTO: La tua Reward Densa vs Logica CSV Densa")
    print("=" * 95)

    try:
        cfg = RewardConfig()
        env = make_env(reward_config=cfg)
        env.reset(seed=SEED)

        dir_map_inv = {0: "est", 1: "sud", 2: "west", 3: "north"}
        action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

        grid = env.unwrapped.grid
        key_pos = door_pos = goal_pos = None
        for i in range(grid.width):
            for j in range(grid.height):
                obj = grid.get(i, j)
                if obj is not None:
                    if obj.type == "key":
                        key_pos = (i, j)
                    elif obj.type == "door":
                        door_pos = (i, j)
                    elif obj.type == "goal":
                        goal_pos = (i, j)

        print(
            f"{'Step':<5} | {'Azione':<8} | {'Stage':<10} | {'Tua Reward (Live)':<20} | {'Reward CSV':<20} | {'Match':<5}"
        )
        print("-" * 95)

        np.random.seed(SEED)
        for step in range(20):  # Facciamo 20 step per avere un buon campione
            base = env.unwrapped
            ax, ay = base.agent_pos
            d_idx = base.agent_dir

            has_key = base.carrying is not None
            is_door_open = False
            if door_pos and grid.get(*door_pos):
                is_door_open = grid.get(*door_pos).is_open

            if has_key and is_door_open:
                stage = "door_open"
                target_pos = goal_pos
            elif has_key:
                stage = "has_key"
                target_pos = door_pos
            else:
                stage = "no_key"
                target_pos = key_pos

            action = np.random.randint(0, 6)  # Evitiamo 'done' per non terminare subito

            # Esegui l'azione sul tuo ambiente reale e prendi la TUA reward
            obs, sys_reward, term, trunc, info = env.step(action)

            # Costruisci la riga fittizia per farci i conti col CSV
            row = {
                "agent_x": ax,
                "agent_y": ay,
                "agent_dir": dir_map_inv[d_idx],
                "goal_pos": str(target_pos),
                "goal_name": stage,
                "action": action_names[action],
            }

            # Calcola cosa avrebbe dato il CSV
            csv_reward = calculate_reward(row)

            match = "SI" if abs(sys_reward - csv_reward) < 0.001 else "NO"
            print(
                f"{step+1:<5} | {action_names[action]:<8} | {stage:<10} | {sys_reward:<20.4f} | {csv_reward:<20.4f} | {match:<5}"
            )

            if term or trunc:
                env.reset()

        env.close()
    except Exception as e:
        print(f"Errore durante il confronto live: {e}")
    print("=" * 95 + "\n")


# ==========================================
# 3. WRAPPER CHE APPLICA LA REWARD DEL CSV
# ==========================================
class CSVLogicRewardWrapper(gym.Wrapper):
    """
    Wrapper che applica esattamente la logica della reward salvata nel CSV.
    Permette di addestrare un agente usando i valori del CSV.
    """

    def __init__(self, env):
        super().__init__(env)
        self.dir_map_inv = {0: "est", 1: "sud", 2: "west", 3: "north"}
        self.action_names = [
            "left",
            "right",
            "forward",
            "pickup",
            "drop",
            "toggle",
            "done",
        ]

    def step(self, action):
        base = self.unwrapped
        ax, ay = base.agent_pos
        d_idx = base.agent_dir

        # Trova target e stage attuali PRIMA di fare lo step
        grid = base.grid
        key_pos = door_pos = goal_pos = None
        for i in range(grid.width):
            for j in range(grid.height):
                obj = grid.get(i, j)
                if obj is not None:
                    if obj.type == "key":
                        key_pos = (i, j)
                    elif obj.type == "door":
                        door_pos = (i, j)
                    elif obj.type == "goal":
                        goal_pos = (i, j)

        has_key = base.carrying is not None
        is_door_open = False
        if door_pos:
            door_obj = grid.get(*door_pos)
            if door_obj and door_obj.is_open:
                is_door_open = True

        if has_key and is_door_open:
            stage = "door_open"
            target = goal_pos
        elif has_key:
            stage = "has_key"
            target = door_pos
        else:
            stage = "no_key"
            target = key_pos

        # Esegui l'azione nell'ambiente base
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Costruisci la riga per calcolare la reward del CSV
        row = {
            "agent_x": ax,
            "agent_y": ay,
            "agent_dir": self.dir_map_inv[d_idx],
            "goal_pos": str(target),
            "goal_name": stage,
            "action": self.action_names[action],
        }
        csv_reward = calculate_reward(row)

        # Se raggiunge il goal, manteniamo il reward finale dell'ambiente (1 - 0.9*...)
        if terminated and reward > 0:
            csv_reward = reward

        return obs, csv_reward, terminated, truncated, info


# ==========================================
# 4. AGENTE Q-LEARNING E TRAINER
# ==========================================
class StateEncoder:
    def encode(self, env, info):
        base = env.unwrapped
        ax, ay = base.agent_pos
        d = base.agent_dir

        stage = info.get("stage", "no_key")
        if hasattr(stage, "value"):
            stage = stage.value
        if stage not in ["no_key", "has_key", "door_open", "goal_reached"]:
            if base.carrying is not None:
                stage = "has_key"
            else:
                stage = "no_key"

        grid = base.grid
        key_pos, door_pos, goal_pos = None, None, None
        for i in range(grid.width):
            for j in range(grid.height):
                obj = grid.get(i, j)
                if obj is not None:
                    if obj.type == "key":
                        key_pos = (i, j)
                    elif obj.type == "door":
                        door_pos = (i, j)
                    elif obj.type == "goal":
                        goal_pos = (i, j)

        if stage == "no_key":
            target_pos = key_pos
        elif stage == "has_key":
            target_pos = door_pos
        elif stage == "door_open":
            target_pos = goal_pos
        else:
            target_pos = (ax, ay)

        tx, ty = target_pos if target_pos else (ax, ay)
        stage_map = {"no_key": 0, "has_key": 1, "door_open": 2, "goal_reached": 3}
        return (tx - ax, ty - ay, d, stage_map.get(stage, 0))


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
        self.q[s][a] += self.alpha * (r + self.gamma * best_next - self.q[s][a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, episodes=2000, max_steps=300):
    agent = QLearningAgent()
    encoder = StateEncoder()
    rewards, successes = [], []
    success_buffer = deque(maxlen=100)

    for ep in range(episodes):
        obs, info = env.reset(seed=SEED if ep == 0 else None)
        state = encoder.encode(env, info)
        ep_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            obs_next, reward, terminated, truncated, info_next = env.step(action)
            done = terminated or truncated
            next_state = encoder.encode(env, info_next)
            agent.update(state, action, reward, next_state, done)
            state, ep_reward, info = next_state, ep_reward + reward, info_next
            if done:
                break

        agent.decay_epsilon()
        rewards.append(ep_reward)

        final_stage = info.get("stage", "unknown")
        if hasattr(final_stage, "value"):
            final_stage = final_stage.value

        is_success = 1 if (final_stage == "goal_reached" or reward > 0.5) else 0
        success_buffer.append(is_success)
        successes.append(np.mean(success_buffer))

    return rewards, successes


# ==========================================
# 5. ESECUZIONE, TRAINING E GRAFICI
# ==========================================
def main():
    # 1. Genera il CSV con la reward densa
    generate_dataset()

    # 2. Confronta a video le due reward dense
    confronta_due_reward_dense()

    episodes = 2000

    # 3. Training Agente 1: Tuo Reward System Denso
    print(f"[3] Inizio Training: Tuo Reward System Denso per {episodes} episodi...")
    cfg = RewardConfig()
    env_native = make_env(reward_config=cfg)
    r_native, s_native = train_agent(env_native, episodes)
    env_native.close()
    print("    Training Native completato.")

    # 4. Training Agente 2: Logica CSV Denso
    print(f"\n[4] Inizio Training: Logica CSV Denso per {episodes} episodi...")
    env_csv = CSVLogicRewardWrapper(DoorKeyEnv(size=6, max_steps=300))
    r_csv, s_csv = train_agent(env_csv, episodes)
    env_csv.close()
    print("    Training CSV completato.")

    # 5. Plotting: Confronto tra le due reward dense
    print("\n[5] Generazione Grafici in corso...")
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(r_native, alpha=0.3, color="blue")
    plt.plot(r_csv, alpha=0.3, color="green")
    plt.plot(
        pd.Series(r_native).rolling(50).mean(),
        color="blue",
        label="Tuo Sistema (MA 50)",
    )
    plt.plot(
        pd.Series(r_csv).rolling(50).mean(), color="green", label="Logica CSV (MA 50)"
    )
    plt.title("Reward Cumulativo per Episodio (Densa vs Densa)")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(s_native, color="blue", label="Tuo Sistema Denso")
    plt.plot(s_csv, color="green", label="Logica CSV Denso")
    plt.title("Success Rate (Media Mobile 100 ep)")
    plt.xlabel("Episodio")
    plt.ylabel("Success Rate %")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("confronto_due_dense.png")
    print("    Grafici salvati in 'confronto_due_dense.png'")
    print("\nFinito!")


if __name__ == "__main__":
    main()

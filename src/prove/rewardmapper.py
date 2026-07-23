#!/usr/bin/env python3
#
# rewardmapper.py
#
# DESIGN
# ------
# Genera un dataset CSV di transizioni (stato, azione, reward) per il
# dominio MiniGrid DoorKey 6×6 usando il reward shaping denso di
# DoorKeyRewardSystem.  Per ognuno dei 3 seed si esegue una camminata
# casuale registrando la tripletta (pos_agente, dir_agente, goal_corrente,
# azione, reward).  Alla fine si ritagliano due subset: uno da 300 righe
# (casuale) e uno da 1000 righe con reward azzerata, utili per
# esperimenti di controllo (abbiamo bisogno di esemplari con reward nulla
# per isolare l'effetto del segnale di reward negli esperimenti).
#
# La scelta di 2000 passi per seed (6000 totali) è un euristico: con 3
# layout diversi e azioni casuali si copre una frazione significativa
# dello spazio degli stati senza pretesa di esaustività.  Per copertura
# totale servirebbe una BFS con snapshot dello stato, ma non è necessario
# per gli esperimenti target.

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from env.rewardsystem import DoorKeyRewardSystem, RewardConfig

ACTIONS = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]
SEEDS = [1, 2, 3]
STEPS_PER_SEED = 2000
OUT_DIR = Path(__file__).parent


def get_goal_pos(unwrapped, stage):
    """Restituisce le coordinate (x,y) dell'obiettivo corrente in base
    allo stage, oppure None se lo stage è goal_reached (non c'è un
    sotto-obiettivo da inseguire).  Serve per sapere verso quale oggetto
    l'agente si sta muovendo nella fase corrente."""
    target_type = {"no_key": "key", "has_key": "door", "door_open": "goal", "goal_reached": None}
    t = target_type.get(stage)
    if t is None:
        return None
    grid = unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == t:
                return (x, y)
    return None


def main():
    random.seed(42)
    rows = []

    # Esplorazione casuale per ogni seed
    for seed in SEEDS:
        env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=None)
        env = DoorKeyRewardSystem(env, RewardConfig())
        obs, info = env.reset(seed=seed)

        for _ in range(STEPS_PER_SEED):
            base = env.unwrapped
            agent_x, agent_y = base.agent_pos
            agent_dir = base.agent_dir
            dir_map = {0: "est", 1: "sud", 2: "west", 3: "north"}
            agent_dir_str = dir_map.get(agent_dir, str(agent_dir))
            stage = info.get("stage", "no_key")
            goal_pos = get_goal_pos(base, stage)

            action = random.randint(0, 6)
            action_name = ACTIONS[action]
            obs, reward, terminated, truncated, info = env.step(action)

            rows.append([agent_x, agent_y, agent_dir_str, str(goal_pos), stage, action_name, round(reward, 6)])

            # Riavviamo con lo stesso seed per mantenere lo stesso layout
            # quando l'episodio termina; in questo modo tutte le transizioni
            # per quel seed condividono la stessa mappa.
            if terminated or truncated:
                obs, info = env.reset(seed=seed)

        env.close()

    # Scrittura del dataset completo
    header = ["agent_x", "agent_y", "agent_dir", "goal_pos", "goal_name", "action", "reward"]
    main_path = OUT_DIR / "reward_mapping.csv"
    with open(main_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Main CSV: {len(rows)} righe -> {main_path}")

    # Subset casuale da 300 righe
    random.shuffle(rows)
    subset300 = rows[:300]
    sub300_path = OUT_DIR / "reward_subset_300.csv"
    with open(sub300_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(subset300)
    print(f"Subset 300: {len(subset300)} righe -> {sub300_path}")

    # Subset seed 1 con reward azzerata
    # ponytail: slice basato sull'ordine di generazione (SEEDS[0], STEPS_PER_SEED=2000)
    seed1z = [row[:] for row in rows[:2000]]
    for row in seed1z:
        row[-1] = 0.0
    seed1z_path = OUT_DIR / "reward_subset_seed1_zero.csv"
    with open(seed1z_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(seed1z)
    print(f"Subset seed 1 zero-reward: {len(seed1z)} righe -> {seed1z_path}")


if __name__ == "__main__":
    main()

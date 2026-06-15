#!/usr/bin/env python3
import csv
import sys
import copy
import gc
from collections import deque
from pathlib import Path

# Aggiunge la cartella genitore al path per importare i moduli locali
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import cast
import gymnasium as gym

# Importazioni di MiniGrid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Door, Key, Goal
from minigrid.core.mission import MissionSpace

# Importazioni dei tuoi moduli
from env.rewardsystem import DoorKeyRewardSystem, RewardConfig
from env import doorkey_events as doorev


# ==========================================
# 1. AMBIENTE CUSTOM ESATTO
# ==========================================
class ExactDoorKeyEnv(MiniGridEnv):
    """
    Ambiente custom per forzare coordinate esatte di muri, porta, oggetti e agente.
    """

    def __init__(
        self, wall_x, door_y, key_pos, goal_pos, agent_pos, agent_dir, size=6, **kwargs
    ):
        self.wall_x = wall_x
        self.door_y = door_y
        self.forced_key_pos = key_pos
        self.forced_goal_pos = goal_pos
        self.forced_agent_pos = agent_pos
        self.forced_agent_dir = agent_dir

        mission_space = MissionSpace(
            mission_func=lambda: "use the key to open the door and then get to the goal"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=1000,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Disegna il muro divisorio verticale
        for i in range(1, height - 1):
            self.grid.set(self.wall_x, i, Wall())

        # Piazza la porta chiusa a chiave
        self.grid.set(self.wall_x, self.door_y, Door("yellow", is_locked=True))

        # Piazza Chiave e Goal
        self.put_obj(Key("yellow"), *self.forced_key_pos)
        self.put_obj(Goal(), *self.forced_goal_pos)

        # Piazza l'agente
        self.agent_pos = self.forced_agent_pos
        self.agent_dir = self.forced_agent_dir


# ==========================================
# 2. FUNZIONI DI SUPPORTO
# ==========================================
ACTIONS = {0: "LEFT", 1: "RIGHT", 2: "FORWARD", 3: "PICKUP", 4: "DROP", 5: "TOGGLE"}


def get_state_hash(env):
    """Genera una firma univoca per lo stato per evitare loop infiniti."""
    base = cast(MiniGridEnv, env.unwrapped)
    stage_obj = env.get_wrapper_attr("curr_stage")
    stage = stage_obj.value if stage_obj else "unknown"

    return (
        tuple(base.agent_pos),
        base.agent_dir,
        env.get_wrapper_attr("key_pos") if not doorev.has_key(env) else "pocket",
        doorev.door_is_open(env),
        stage,
    )


# ==========================================
# 3. I 3 SEED HARDCODED
# ==========================================
# Un mix rappresentativo: muri a x=2 e x=3, porte in punti diversi, e varie direzioni iniziali
SEEDS = [
    {
        "wall_x": 2,
        "door_y": 2,
        "key_pos": (1, 1),
        "goal_pos": (3, 3),
        "agent_pos": (1, 4),
        "agent_dir": 0,
    },
]


# ==========================================
# 4. CICLO PRINCIPALE: IL MULTIVERSO RIDOTTO
# ==========================================
def main():
    print("🚀 Avvio Mappatura (Versione a 3 Seed)...")
    cfg = RewardConfig()

    csv_file = "doorkey_sampled_multiverse_qlearning.csv"
    headers = [
        "seed_idx",
        "obs_agent_x",
        "obs_agent_y",
        "obs_agent_dir",
        "obs_key_pos",
        "obs_door_open",
        "obs_stage",
        "action_taken",
        "reward_obtained",
    ]

    with open(csv_file, mode="w", newline="", buffering=1) as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        total_rows = 0

        for i, layout in enumerate(SEEDS, start=1):
            wall_x = layout["wall_x"]
            door_y = layout["door_y"]
            key_pos = layout["key_pos"]
            goal_pos = layout["goal_pos"]
            agent_pos = layout["agent_pos"]
            agent_dir = layout["agent_dir"]

            print(
                f"Esplorazione Seed {i}/3 | Muro: {wall_x},{door_y} | "
                f"Goal: {goal_pos} | Chiave: {key_pos} | "
                f"Agente: {agent_pos} Dir: {agent_dir}"
            )

            # Creiamo l'universo specifico
            base_env = ExactDoorKeyEnv(
                wall_x=wall_x,
                door_y=door_y,
                key_pos=key_pos,
                goal_pos=goal_pos,
                agent_pos=agent_pos,
                agent_dir=agent_dir,
                size=6,
            )

            env = DoorKeyRewardSystem(base_env, cfg)
            env.reset()

            queue = deque([env])
            visited = set([get_state_hash(env)])

            # Esplorazione BFS
            while queue:
                curr_env = queue.popleft()
                parent_hash = get_state_hash(curr_env)

                unwrapped = cast(MiniGridEnv, curr_env.unwrapped)
                px, py = unwrapped.agent_pos
                pdir = unwrapped.agent_dir

                pstage_obj = curr_env.get_wrapper_attr("curr_stage")
                pstage = pstage_obj.value if pstage_obj else "unknown"

                for action_idx, action_name in ACTIONS.items():
                    branch_env = copy.deepcopy(curr_env)
                    obs, reward, term, trunc, info = branch_env.step(action_idx)

                    new_hash = get_state_hash(branch_env)

                    b_unwrapped = cast(MiniGridEnv, branch_env.unwrapped)
                    cx, cy = b_unwrapped.agent_pos
                    cdir = b_unwrapped.agent_dir

                    cstage_obj = branch_env.get_wrapper_attr("curr_stage")
                    cstage = cstage_obj.value if cstage_obj else "unknown"

                    events = info.get("events", {})
                    rb = info.get("reward_breakdown", {})

                    # parent_hash = (agent_pos, agent_dir, key_pos, door_open, stage)
                    # Scrittura nel CSV
                    writer.writerow(
                        [
                            i,
                            parent_hash[0][0],
                            parent_hash[0][1],
                            parent_hash[1],
                            str(parent_hash[2]),
                            parent_hash[3],
                            parent_hash[4],
                            action_name,
                            round(reward, 4),
                        ]
                    )
                    total_rows += 1

                    if total_rows % 5000 == 0:
                        file.flush()

                    # Aggiungi i nuovi stati alla coda
                    if new_hash not in visited and not term and not trunc:
                        visited.add(new_hash)
                        queue.append(branch_env)
                    else:
                        del branch_env

                # Pulizia dell'ambiente "padre" esplorato
                del curr_env

            # Pulizia per il prossimo seed
            queue.clear()
            visited.clear()
            del env
            del base_env
            gc.collect()

    print(
        f"\n🎉 Mappatura completata! Generate {total_rows} transizioni nel file {csv_file}"
    )

    import random

    print("\nGenerazione del subset random (senza seed)...")
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        seed_idx_col = header.index("seed_idx")

        new_header = [h for i, h in enumerate(header) if i != seed_idx_col]
        new_reward_idx_col = new_header.index("reward_obtained")

        rows = list(reader)
        processed_rows = [
            [val for i, val in enumerate(r) if i != seed_idx_col] for r in rows
        ]

    subset_size = 500

    subset_rows = random.sample(processed_rows, subset_size)

    subset_file = "doorkey_sampled_subset.csv"
    with open(subset_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(subset_rows)

    print(f"Salvato subset di {len(subset_rows)} transizioni in {subset_file}")

    print("\nGenerazione del dataset completo con reward 0...")
    file_zero_reward = "doorkey_sampled_zero_reward.csv"

    rows_zero = []
    for r in processed_rows:
        new_r = list(r)
        new_r[new_reward_idx_col] = 0
        rows_zero.append(new_r)

    with open(file_zero_reward, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(rows_zero)

    print(f"Salvato dataset completo con reward a 0 in {file_zero_reward}")


if __name__ == "__main__":
    main()

import os
import re
import torch
from ctypes import string_at
from enum import IntEnum
from typing import cast
from gymnasium.spaces import Dict, Box, Discrete
from PIL import Image
import gymnasium as gym
import numpy as np
from collections import deque
import re
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper
from minigrid.manual_control import ManualControl
from enum import IntEnum
from monorepo import GroqLLM, load_api_keys, GROQ_MULTIMODAL_MODEL_ID
from dotenv import load_dotenv
import ollama

load_api_keys()
client = GroqLLM(model_id="openai/gpt-oss-120b")


PROMPT = """ROLE
You are the eye and brain of an agent in an unknown 2D GridWorld environment.
YOUR GOAL IS TO UNDERSTAND THE LEVEL'S RULES, SEVERELY EVALUATE THE PREVIOUS ACTION, AND PRODUCE EXACTLY THE REQUESTED FORMAT AT THE END.**

AGENT OBSERVATIONS:<osservazione>

UNIVERSAL RULES
    Walls cannot be crossed. Hitting them is a serious error.
    The agent moves one cell at a time OR rotates in place.
    The agent interacts only with the cell in front of it (direction it's facing).
    If the agent doesn't have the key in its pocket, it MUST pick it up first.
    if you dont see a useful object on the grid suppose that is in the agent's pocket.

HOW TO DEDUCE THE ENVIRONMENT AND CURRENT TARGET
FINAL GOAL: The GREEN square.
OBSTACLE LOGIC: If the path is blocked (e.g., door of color X closed), the CURRENT TARGET is not the door, but the object (e.g., key of color X) needed to open it. Only after picking it up does the door become the target.

AVAILABLE ACTIONS:
0: LEFT, 1: RIGHT, 2: FORWARD, 3: PICKUP (Collect), 4: DROP (Drop), 5: TOGGLE (Use/Open).

ANALYSIS INSTRUCTIONS (Follow steps in strict order)
Step 1 (Target Identification): What is the agent's CURRENT TARGET at this exact moment? (e.g., The red key? The red door? The green goal?).
Step 2 (Previous Action Analysis): What action did the agent just perform? (0-5).
Step 3 (Spatial and Orientation Analysis - CRITICAL):
    What is the agent facing?
    Did the agent picked up anything?
    Did the agent have anything in it's poket?
    Did the previous action bring the agent closer to the CURRENT TARGET?
    Did the agent move away?
    Did the agent turn toward a useless wall or with its back to the target?
    The agent seems trying to turn torward the CURRENT GOAL?
    Did it try to walk into a wall or pick up nothing?
    What is the agent going to do from here?
Step 3 (Estimate): Looking at the current state, the last action helped the agent to increase its progress?.
Step 4 (Reward Calculation): Based on the REWARD CRITERIA below, what is the exact numerical reward for this action, watch the last action and think about what action may the agent will do from its new position or orientation?
Step 5 (Estimate Progress): Looking at the current state, estimate the agent's global progress in completing the puzzle (0.0 to 100.0).

REWARD CRITERIA (FOLLOW TO THE LETTER)
Assign the reward evaluating the action quality based on these strict rules:

    Very positive: Final goal reached.
    Positive: Vital sub-goal achieved (e.g., just picked up the right key or opened the door).
    Slightly positive: Moved physically closer (Manhattan Distance) to the CURRENT TARGET or the action is useful to face the agent to the objective.
    Slightly negative: TIME PENALTY. Apply it always for neutral actions to punish wasting time.
    Negative: DISTANCING. Assign SEVERELY if the agent steps back from the target or turns (LEFT/RIGHT) away from the useful direction to the CURRENT TARGET.
    Very negative: STUPID ACTION. Hit a wall, tried TOGGLE on empty space, or DROPPED a useful object.
    Remember the reward MUST be between -1.0 and 1.0, so scale accordingly.

RESPONSE INSTRUCTIONS 
You must strictly follow this XML format. Put your step-by-step reasoning inside the <analysis> tags. Then, print ONLY the two decimal values separated by a comma inside the <metrics> tags. NO extra text outside these tags.

<analysis>
[Write your Step 1 to 5 reasoning here]
</analysis>
<metrics>reward, progress</metrics>"""


# ==========================================
# CANALE 0: Tipi di Oggetti
# ==========================================
class ObjectType(IntEnum):
    UNSEEN = 0
    EMPTY = 1
    WALL = 2
    FLOOR = 3
    DOOR = 4
    KEY = 5
    BALL = 6
    BOX = 7
    GOAL = 8
    LAVA = 9
    AGENT = 10


# ==========================================
# CANALE 1: Colori
# ==========================================
class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5


# ==========================================
# CANALE 2: Stati (Principalmente per le porte)
# ==========================================
class DoorState(IntEnum):
    OPEN = 0
    CLOSED = 1
    LOCKED = 2


# --- DIZIONARI DI TRADUZIONE PER IL PROMPT VLM ---
# Utili per convertire l'Enum in parole italiane comprensibili per il tuo LLM
TRANSLATE_OBJ = {
    ObjectType.EMPTY: "empty",
    ObjectType.WALL: "wall",
    ObjectType.DOOR: "door",
    ObjectType.KEY: "key",
    ObjectType.BALL: "ball",
    ObjectType.BOX: "box",
    ObjectType.GOAL: "goal",
    ObjectType.LAVA: "lava",
}

TRANSLATE_COLOR = {
    Color.RED: "red",
    Color.GREEN: "green",
    Color.BLUE: "blue",
    Color.PURPLE: "purple",
    Color.YELLOW: "yellow",
    Color.GREY: "grey",
}


class VLMDebugWrapper(gym.Wrapper):
    def __init__(self, env, query_every: int = 1):
        super().__init__(env)
        self.query_every = query_every
        self._step_count = 0
        self.cache = deque(maxlen=2)

        self._last_action = 0
        self._last_env_reward = 0.0
        self._last_obs = None
        self.last_progress = 0.0

        self.reset()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self.cache.clear()

        self._last_obs = obs
        return obs, info

    def step(self, action):
        past_grid = self._generate_grid_csv(self._last_obs)

        obs, env_reward, terminated, truncated, info = self.env.step(action)

        current_grid = self._generate_grid_csv(obs)

        self._last_obs = obs
        self._step_count += 1
        self._last_action = action
        self._last_env_reward = env_reward

        comparative_grid = self._stack_grids(past_grid, current_grid)

        prompt = PROMPT.replace("<osservazione>", comparative_grid)
        reward, progress = self._get_reward(prompt)

        reward += float(env_reward)
        self.last_progress = progress

        return obs, reward, terminated, truncated, info

    def _stack_grids(self, past_grid: str, current_grid: str) -> str:
        """
        Places the past state grid above the current state grid.
        """
        stacked_output = "--- PAST STATE ---\n"
        stacked_output += past_grid
        stacked_output += "--- CURRENT STATE ---\n"
        stacked_output += current_grid
        return stacked_output

    def _generate_grid_csv(self, obs) -> str:
        from minigrid.minigrid_env import (
            MiniGridEnv,
        )  # Assicurati di averlo importato in alto nel file

        grid = obs["image"]
        width, height, channels = grid.shape

        csv_output = ""

        DIRECTION_MAP = {
            0: "facing_right",
            1: "facing_down",
            2: "facing_left",
            3: "facing_up",
        }

        # Get the real agent position from the unwrapped environment
        unwrapped_env = cast(MiniGridEnv, self.env.unwrapped)
        agent_x, agent_y = unwrapped_env.agent_pos
        agent_dir = unwrapped_env.agent_dir

        for y in range(height):
            row_cells = []

            for x in range(width):
                # 1. Check for Agent
                if x == agent_x and y == agent_y:
                    dir_text = DIRECTION_MAP.get(agent_dir, "unknown")
                    row_cells.append(f"agent_{dir_text}")
                    continue

                # 2. Extract Channels
                obj_id = grid[x, y, 0]
                color_id = grid[x, y, 1]
                state_id = grid[x, y, 2]

                # 3. Handle basic objects
                if obj_id == ObjectType.WALL:
                    row_cells.append("wall")
                    continue

                if obj_id == ObjectType.EMPTY:
                    row_cells.append("empty")
                    continue

                # 4. Dynamic translation
                try:
                    obj_name = TRANSLATE_OBJ.get(ObjectType(obj_id), f"obj_{obj_id}")
                    color_name = TRANSLATE_COLOR.get(Color(color_id), f"col_{color_id}")
                except ValueError:
                    obj_name = f"unknown_{obj_id}"
                    color_name = f"unknown_{color_id}"

                cell_string = f"{obj_name}_{color_name}"

                # 5. Handle specific states (e.g., Doors)
                if obj_id == ObjectType.DOOR:
                    if state_id == DoorState.OPEN:
                        cell_string += "_open"
                    elif state_id == DoorState.CLOSED:
                        cell_string += "_closed"
                    elif state_id == DoorState.LOCKED:
                        cell_string += "_locked"

                elif obj_id == ObjectType.AGENT:
                    dir_text = DIRECTION_MAP.get(agent_dir, "unknown")
                    cell_string = f"agent_{dir_text}"

                # Append the constructed cell string to the row list
                row_cells.append(cell_string)

            # Join all cells in the row with a comma and add a newline
            csv_output += ",".join(row_cells) + ";\n"

        return csv_output

    def _get_reward(self, prompt):
        import time

        try:

            response = ollama.chat(
                model="nemotron-3-nano:4b",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "num_ctx": 4096,
                },
            )
            content = response["message"]["content"]

            # content = client.ask(prompt=prompt)
            pattern = (
                r"<metrics>\s*([+-]?\d+[.,]\d+)\s*,\s*([+-]?\d+[.,]\d+)\s*</metrics>"
            )

            match = re.search(pattern, content, re.DOTALL)
            print(content)

            if match:
                # Sostituiamo la virgola col punto per rendere float() felice
                v1 = float(match.group(1).replace(",", "."))
                v2 = float(match.group(2).replace(",", "."))

                return v1, v2
            else:
                print("⚠️ Formato <metrics> non trovato nella risposta dell'AI")
                print(f"Risposta AI: {content}")
                return 0.0, 0.0

        except Exception as e:
            print(f"[❌ ERRORE CRITICO OLLAMA] Fallimento: {e}")
            return 0.0, 0.0


class CustomManualControl(ManualControl):
    def key_handler(self, event):
        key = event.key

        # Mappatura tasti corretta per MiniGrid Actions
        if key == "escape":
            self.env.close()
        elif key == "backspace":
            self.reset()
        elif key == "left":
            self.step(Actions.left)
        elif key == "right":
            self.step(Actions.right)
        elif key == "up":
            self.step(Actions.forward)
        elif key == "space":
            self.step(Actions.toggle)
        elif key == "p":
            self.step(Actions.pickup)
        elif key == "d":
            self.step(Actions.drop)
        else:
            # Gestione default per altri tasti
            super().key_handler(event)


def main():
    print("Creazione ambiente MiniGrid...")
    # 'human' permette la visualizzazione e l'input da tastiera
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
    env = FullyObsWrapper(env)
    env = VLMDebugWrapper(env, query_every=1)

    print("\n" + "*" * 60)
    print("CONTROLLI MANUALI ATTIVI:")
    print("Freccia SU:    Avanti")
    print("Freccia SX/DX: Ruota")
    print("Barra Spazio:  Apri (Toggle)")
    print("Tasto 'P':     Raccogli (Pickup)")
    print("Tasto 'D':     Lascia (Drop)")
    print("*" * 60 + "\n")

    manual_control = CustomManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    main()

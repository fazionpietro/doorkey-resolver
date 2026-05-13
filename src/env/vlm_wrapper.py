import re
import time
import requests
from enum import IntEnum
from typing import cast

import gymnasium as gym
import numpy as np
from minigrid.minigrid_env import MiniGridEnv


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


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    PURPLE = 3
    YELLOW = 4
    GREY = 5


class DoorState(IntEnum):
    OPEN = 0
    CLOSED = 1
    LOCKED = 2


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


class VLMDebugWrapper(gym.Wrapper):
    OLLAMA_MODEL = "nemotron-3-nano:4b"
    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_TIMEOUT = 10.0

    _DIRECTION_MAP = {
        0: "facing_right",
        1: "facing_down",
        2: "facing_left",
        3: "facing_up",
    }

    _METRICS_RE = re.compile(
        r"<metrics>\s*([+-]?\d+[.,]\d+)\s*,\s*([+-]?\d+[.,]\d+)\s*</metrics>",
        re.DOTALL,
    )

    def __init__(self, env: gym.Env, query_every: int = 10):
        super().__init__(env)
        self.query_every = query_every
        self._step_count = 0
        self._last_obs = None
        self.last_progress = 0.0
        self.last_vlm_reward = 0.0
        self.reset()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._last_obs = obs
        self.last_vlm_reward = 0.0
        self.last_progress = 0.0
        return obs, info

    def step(self, action, ignore=False):
        past_grid = self._generate_grid_csv(self._last_obs)
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        current_grid = self._generate_grid_csv(obs)

        self._last_obs = obs
        self._step_count += 1

        if self._step_count % self.query_every == 0 and ignore == False:
            comparative = self._stack_grids(past_grid, current_grid)
            prompt = PROMPT.replace("<osservazione>", comparative)
            vlm_reward, progress = self._query_vlm(prompt)
            self.last_progress = progress
            self.last_vlm_reward = vlm_reward
        else:
            vlm_reward = self.last_vlm_reward

        total_reward = vlm_reward + float(env_reward)
        return obs, total_reward, terminated, truncated, info

    def _stack_grids(self, past: str, current: str) -> str:
        return f"--- PAST STATE ---\n{past}--- CURRENT STATE ---\n{current}"

    def _generate_grid_csv(self, obs) -> str:
        grid = obs["image"]
        width, height, _ = grid.shape
        unwrapped = cast(MiniGridEnv, self.env.unwrapped)
        agent_x, agent_y = unwrapped.agent_pos
        agent_dir = unwrapped.agent_dir

        rows = []
        for y in range(height):
            cells = []
            for x in range(width):
                if x == agent_x and y == agent_y:
                    dir_text = self._DIRECTION_MAP.get(agent_dir, "unknown")
                    cells.append(f"agent_{dir_text}")
                    continue

                obj_id = int(grid[x, y, 0])
                color_id = int(grid[x, y, 1])
                state_id = int(grid[x, y, 2])

                if obj_id == ObjectType.WALL:
                    cells.append("wall")
                    continue
                if obj_id == ObjectType.EMPTY:
                    cells.append("empty")
                    continue

                try:
                    obj_name = TRANSLATE_OBJ.get(ObjectType(obj_id), f"obj_{obj_id}")
                    color_name = TRANSLATE_COLOR.get(Color(color_id), f"col_{color_id}")
                except ValueError:
                    obj_name = f"unknown_{obj_id}"
                    color_name = f"unknown_{color_id}"

                cell = f"{obj_name}_{color_name}"

                if obj_id == ObjectType.DOOR:
                    if state_id == DoorState.OPEN:
                        cell += "_open"
                    elif state_id == DoorState.CLOSED:
                        cell += "_closed"
                    elif state_id == DoorState.LOCKED:
                        cell += "_locked"

                cells.append(cell)
            rows.append(",".join(cells))

        return ";\n".join(rows) + ";\n"

    def _query_vlm(self, prompt: str) -> tuple[float, float]:
        payload = {
            "model": self.OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.0,
                "num_keep": 0,
                "num_ctx": 4096,
            },
        }

        try:
            response = requests.post(self.OLLAMA_URL, json=payload, timeout=30.0)
            response.raise_for_status()

            raw_response = response.json()
            content = raw_response.get("message", {}).get("content", "").strip()

            if not content:
                print(
                    f"[❌ VLM ERROR] Ollama ha restituito un contenuto vuoto! JSON: {raw_response}"
                )
                return 0.0, 0.0

            # Cerca i valori
            match = self._METRICS_RE.search(content)

            if match:
                v1 = float(match.group(1).replace(",", "."))
                v2 = float(match.group(2).replace(",", "."))
                print(v1, v2)
                return v1, v2
            else:
                # Se l'LLM ha risposto ma non ha formattato bene, vediamo cosa ha scritto
                print(f"[⚠️ VLM WARN] Regex fallita. Risposta dell'LLM:\n{content}")

        except requests.exceptions.Timeout:
            print(f"[⏱️ VLM TIMEOUT] Ollama non ha risposto entro 30 secondi.")
        except requests.exceptions.ConnectionError:
            print(
                f"[🔌 VLM CONN ERROR] Impossibile connettersi a {self.OLLAMA_URL}. Ollama è acceso?"
            )
        except Exception as e:
            print(f"[❌ VLM ERROR] Eccezione inaspettata: {e}")

        # Fallback in caso di qualsiasi errore
        return 0.0, 0.0

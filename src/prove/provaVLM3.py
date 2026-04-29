import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym
import numpy as np
from collections import deque
import re
from qwen_vl_utils import process_vision_info
from minigrid.wrappers import FullyObsWrapper
from minigrid.manual_control import ManualControl

# ==========================================
# CONFIG
# ==========================================
model_id = "Qwen/Qwen3-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Caricamento modello...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

TILE_SIZE = 64  # frame più grandi = VLM vede meglio

# ==========================================
# PROMPTS DECOMPOSED
# ==========================================

PROMPT_STAGE = """This is a MiniGrid-DoorKey game frame (enlarged for clarity).

Visual guide:
- RED TRIANGLE = agent (tip points in facing direction)
- YELLOW SMALL SQUARE = key (pick it up)
- GREY GRID = walls
- BROWN/DARK RECTANGLE = door (locked or open)
- GREEN SQUARE = goal (reach to win)

What is the current game stage? Answer with EXACTLY one of these labels and nothing else:

SEEKING_KEY    → key is visible on the floor, agent does not hold it
HOLDING_KEY    → agent is carrying the key (key no longer on floor, or shown at agent)
DOOR_OPEN      → door has been opened (passage is clear)
ON_GOAL        → agent stands on the green square

Label:"""

PROMPT_DIRECTION = """You see TWO MiniGrid frames side by side.
LEFT frame = state BEFORE the action.
RIGHT frame = state AFTER the action.
The red triangle is the agent.

Current task stage: {stage}
Current target: {target_description}

Did the agent move CLOSER or FARTHER from the target, or stay in SAME position?

Rules:
- If the agent's triangle moved and is visually nearer to {target_name} → CLOSER
- If the agent's triangle moved and is visually farther from {target_name} → FARTHER  
- If the triangle is in the exact same spot (only rotated, or bumped wall) → SAME

Answer with exactly one word (CLOSER / FARTHER / SAME):"""

STAGE_TO_TARGET = {
    "SEEKING_KEY": ("the yellow key", "key"),
    "HOLDING_KEY": ("the door", "door"),
    "DOOR_OPEN": ("the green goal", "goal"),
    "ON_GOAL": ("the green goal", "goal"),
}

STAGE_PROGRESS = {
    "SEEKING_KEY": 0.0,
    "HOLDING_KEY": 0.33,
    "DOOR_OPEN": 0.66,
    "ON_GOAL": 1.0,
}

MOVEMENT_REWARD = {
    "CLOSER": 0.2,
    "FARTHER": -0.2,
    "SAME": -0.05,  # piccola penalità per stallo/rotazione
}

STAGE_COMPLETION_REWARD = {
    # (prev_stage, curr_stage): reward
    ("SEEKING_KEY", "HOLDING_KEY"): 1.0,  # raccolto key
    ("HOLDING_KEY", "DOOR_OPEN"): 1.0,  # aperta porta
    ("DOOR_OPEN", "ON_GOAL"): 1.0,  # raggiunto goal
}


# ==========================================
# VLM CALLS
# ==========================================


def _call_vlm(image: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
    """Chiamata base al VLM, ritorna stringa grezza."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy — più deterministico
            temperature=None,
            top_p=None,
        )

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return response


def get_stage(frame: Image.Image) -> str:
    """Classifica lo stage corrente dal frame attuale."""
    raw = _call_vlm(frame, PROMPT_STAGE, max_new_tokens=16)
    print(f"   [VLM stage raw]: '{raw}'")
    for label in ["ON_GOAL", "DOOR_OPEN", "HOLDING_KEY", "SEEKING_KEY"]:
        if label in raw.upper():
            return label
    return "SEEKING_KEY"  # fallback


def get_direction(before: Image.Image, after: Image.Image, stage: str) -> str:
    """Classifica se l'agente si è avvicinato/allontanato dal target."""
    target_desc, target_name = STAGE_TO_TARGET.get(stage, ("the target", "target"))
    prompt = PROMPT_DIRECTION.format(
        stage=stage,
        target_description=target_desc,
        target_name=target_name,
    )
    # Affianca i due frame con etichetta
    pair = _make_labeled_pair(before, after)
    raw = _call_vlm(pair, prompt, max_new_tokens=8)
    print(f"   [VLM direction raw]: '{raw}'")
    for label in ["CLOSER", "FARTHER", "SAME"]:
        if label in raw.upper():
            return label
    return "SAME"  # fallback


def _make_labeled_pair(before: Image.Image, after: Image.Image) -> Image.Image:
    """Affianca due frame con etichette LEFT/RIGHT sopra."""
    sep = 8
    label_h = 24
    w = before.width + sep + after.width
    h = max(before.height, after.height) + label_h

    img = Image.new("RGB", (w, h), (240, 240, 240))
    img.paste(before, (0, label_h))
    img.paste(after, (before.width + sep, label_h))

    draw = ImageDraw.Draw(img)
    draw.text((before.width // 2 - 20, 4), "LEFT (before)", fill=(0, 0, 0))
    draw.text(
        (before.width + sep + after.width // 2 - 20, 4), "RIGHT (after)", fill=(0, 0, 0)
    )
    return img


# ==========================================
# WRAPPER
# ==========================================


class VLMRewardWrapper(gym.Wrapper):
    def __init__(self, env, query_every: int = 1, tile_size: int = TILE_SIZE):
        super().__init__(env)
        self.query_every = query_every
        self.tile_size = tile_size
        self._step_count = 0
        self._prev_frame = None
        self._prev_stage = "SEEKING_KEY"
        self._last_metrics = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._prev_frame = self._get_frame()
        self._prev_stage = "SEEKING_KEY"
        return obs, info

    def _get_frame(self) -> Image.Image:
        rgb = self.env.unwrapped.get_frame(highlight=True, tile_size=self.tile_size)
        img = Image.fromarray(np.array(rgb))
        return img

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        curr_frame = self._get_frame()
        self._step_count += 1

        vlm_reward = 0.0
        progress = STAGE_PROGRESS[self._prev_stage]
        curr_stage = self._prev_stage
        direction = "SAME"

        if self._step_count % self.query_every == 0:
            print(f"\n[Step {self._step_count}] VLM analysis...")

            # 1. Classifica stage corrente (solo frame attuale)
            curr_stage = get_stage(curr_frame)
            progress = STAGE_PROGRESS[curr_stage]

            # 2. Controlla se stage è avanzato → reward bonus
            stage_key = (self._prev_stage, curr_stage)
            if curr_stage != self._prev_stage and stage_key in STAGE_COMPLETION_REWARD:
                vlm_reward = STAGE_COMPLETION_REWARD[stage_key]
                print(f"   🎉 Stage completato: {self._prev_stage} → {curr_stage}")
            else:
                # 3. Classifica avvicinamento (due frame)
                if self._prev_frame is not None:
                    direction = get_direction(self._prev_frame, curr_frame, curr_stage)
                    vlm_reward = MOVEMENT_REWARD.get(direction, 0.0)

            self._prev_stage = curr_stage

            self._last_metrics = {
                "vlm_reward": vlm_reward,
                "progress": progress,
                "stage": curr_stage,
                "direction": direction,
            }

            print(f"   Stage:    {curr_stage}")
            print(f"   Direction:{direction}")
            print(f"   Reward:   {vlm_reward:+.2f}")
            print(f"   Progress: {progress:.0%}")

        self._prev_frame = curr_frame
        info.update(self._last_metrics)
        return obs, vlm_reward, terminated, truncated, info


# ==========================================
# MAIN
# ==========================================


class CustomManualControl(ManualControl):
    def key_handler(self, event):
        key = event.key
        actions = self.env.unwrapped.actions
        if key == "escape":
            self.env.close()
        elif key == "backspace":
            self.reset(self.seed)
        elif key == "left":
            self.step(actions.left)
        elif key == "right":
            self.step(actions.right)
        elif key == "up":
            self.step(actions.forward)
        elif key == "space":
            self.step(actions.toggle)
        elif key == "p":
            self.step(actions.pickup)
        elif key == "d":
            self.step(actions.drop)
        else:
            super().key_handler(event)


def main():
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
    env = FullyObsWrapper(env)
    env = VLMRewardWrapper(env, query_every=1, tile_size=TILE_SIZE)

    print("\n" + "*" * 60)
    print("CONTROLLI: SU=avanza  SX/DX=ruota  SPAZIO=usa  P=raccogli  D=lascia")
    print("*" * 60 + "\n")

    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()

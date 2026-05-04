import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
)
from PIL import Image
import gymnasium as gym
import numpy as np
from collections import deque

import re
from minigrid.wrappers import FullyObsWrapper
from minigrid.manual_control import ManualControl

# ==========================================
# CONFIGURAZIONE VLM
# ==========================================

model_id = "google/paligemma-3b-mix-224"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Caricamento del modello in corso... (potrebbe richiedere qualche minuto)")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="~/.env.ml",
)
processor = AutoProcessor.from_pretrained(model_id)

PROMPT = """descrivi l'ambiente"""


# ==========================================
# WRAPPER AMBIENTE
# ==========================================
class VLMDebugWrapper(gym.Wrapper):
    def __init__(self, env, query_every: int = 1):
        super().__init__(env)
        self.query_every = query_every
        self._step_count = 0
        self.cache = deque(maxlen=1)
        self.last_metrics = "0.0, 0.0, 0, 0, 0"

        # Inizializza l'ambiente per ottenere il primo frame
        self.env.reset()
        self._get_frame()
        self._last_action = 0
        self._last_env_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self.cache.clear()
        self._get_frame()
        return obs, info

    def _get_frame(self):
        # minigrid get_frame restituisce sempre l'rgb array a prescindere dal render_mode
        rgb = self.env.unwrapped.get_frame(highlight=True)
        img = Image.fromarray(np.array(rgb))
        self.cache.append(img)
        return img

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._get_frame()
        self._step_count += 1
        self._last_action = action
        self._last_env_reward = env_reward

        if self._step_count % self.query_every == 0 and len(self.cache) == 1:
            print(f"\n[Step {self._step_count}] - Analisi VLM in corso, attendere...")
            stage_id, vlm_r, prog, dist, dx, dy = self._get_reward()

            print(f"📊 METRICHE ESTRATTE:")
            print(f"   Stage ID: {stage_id}")
            print(f"   VLM Reward: {vlm_r}")
            print(f"   Progress: {prog * 100}%")
            print(f"   Distance: {dist} (dx: {dx}, dy: {dy})\n")

        return obs, env_reward, terminated, truncated, info

    def _get_reward(self):
        separator = 10

        width, height = zip(*(img.size for img in self.cache))

        larghezza_totale = sum(width) + (separator * (len(self.cache) - 1))
        altezza_massima = max(height)

        new_img = Image.new("RGB", (larghezza_totale, altezza_massima), "white")

        offset_x = 0
        for i, img in enumerate(self.cache):
            new_img.paste(img, (offset_x, 0))
            offset_x += img.width + separator

        return self.get_vlm_reward(new_img)

    def get_vlm_reward(self, image: Image.Image):

        text = "<image>\n" + PROMPT + "\n\n"

        inputs = processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=2200)

        # Rimuoviamo il prompt dall'output
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

        # --- DEBUG VISIVO: STAMPIAMO IL RAGIONAMENTO DEL MODELLO ---
        print("\n" + "=" * 50)
        print("🧠 RAGIONAMENTO VLM (Chain of Thought):")
        print("-" * 50)
        print(response)
        print("=" * 50 + "\n")

        metrics_match = re.search(r"<METRICS>(.*?)</METRICS>", response, re.DOTALL)
        if metrics_match:
            self.last_metrics = metrics_match.group(1).strip()

        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


# ==========================================
# AVVIO MANUALE (CON CONTROLLI MODIFICATI)
# ==========================================


class CustomManualControl(ManualControl):
    def key_handler(self, event):
        key = event.key

        # Sbucciamo i wrapper per accedere alle azioni originali di MiniGrid!
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
        elif key == "p":  # <-- Il tasto P per raccogliere
            self.step(actions.pickup)
        elif key == "d":
            self.step(actions.drop)
        else:
            # Per qualsiasi altro tasto non mappato
            super().key_handler(event)


def main():
    print("Creazione ambiente MiniGrid...")
    # Render mode 'human' per mostrare a schermo, il wrapper userà get_frame() per il VLM
    env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
    env = FullyObsWrapper(env)
    env = VLMDebugWrapper(env, query_every=1)

    print("\n" + "*" * 60)
    print("CONTROLLI MANUALI AGGIORNATI:")
    print("Freccia SU: Vai avanti")
    print("Freccia SX: Gira a sinistra")
    print("Freccia DX: Gira a destra")
    print("Barra Spaziatrice: Usa / Apri porta")
    print("Tasto 'P': Raccogli oggetto (Chiave)")
    print("Tasto 'D': Lascia oggetto")
    print("*" * 60 + "\n")

    # Avviamo il controllo manuale usando la nostra classe personalizzata
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import torch
import re
import numpy as np
from PIL import Image
import gymnasium as gym
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# INIZIALIZZAZIONE GLOBALE MOONDREAM
# ==========================================
print(f"CUDA disponibile: {torch.cuda.is_available()}")
model_id = "vikhyatk/moondream2"
revisione_stabile = "2024-08-26"
print("Caricamento modello Moondream2 in memoria...")

# AGGIUNGI revision=revisione_stabile a ENTRAMBI i comandi
vlm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision=revisione_stabile,
    torch_dtype=torch.float16,
)
vlm_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revisione_stabile)

device = "cuda" if torch.cuda.is_available() else "cpu"
vlm_model = vlm_model.to(device)  # type: ignore
print("Modello Moondream2 caricato con successo!")

REWARD_PROMPT = """Sei un valutatore di stati per un agente RL in MiniGrid DoorKey 8x8. Devi supporre
quale sia il suo obbiettivo in base all'input.
Osserva l'immagine e assegna un punteggio float da -1.0 a 1.0 in base al progresso dell agente.

RISPONDI SOLO CON UN NUMERO DECIMALE (es: 0.220, 0.5, -0.1). Nessun testo aggiuntivo."""


def get_vlm_reward(frame_after: Image.Image) -> float:
    """Interroga Moondream e restituisce il float."""
    enc_image = vlm_model.encode_image(frame_after)  # type: ignore
    risposta = vlm_model.answer_question(enc_image, REWARD_PROMPT, vlm_tokenizer)  # type: ignore

    match = re.search(r"[-+]?\d*\.\d+|\d+", risposta.strip())
    if match:
        try:
            return max(-1.0, min(1.0, float(match.group())))
        except ValueError:
            return 0.0
    return 0.0


class MoondreamDenseRewardWrapper(gym.Wrapper):
    def __init__(self, env, beta: float = 0.5, query_every: int = 5):
        super().__init__(env)
        self.beta = beta
        self.query_every = query_every
        self._step_count = 0
        self._last_reward = 0.0

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
        return Image.fromarray(np.array(rgb))

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        frame_after = self._get_frame()

        # Interroga il VLM ogni X step
        if self._step_count % self.query_every == 0:
            self._last_reward = get_vlm_reward(frame_after)

        total_reward = float(env_reward) + self.beta * self._last_reward
        self._frame_before = frame_after
        self._step_count += 1

        info["vlm_reward"] = self._last_reward
        info["env_reward"] = env_reward
        info["total_reward"] = total_reward
        return obs, total_reward, terminated, truncated, info

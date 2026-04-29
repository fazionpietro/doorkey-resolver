import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import gymnasium as gym
import numpy as np
from collections import deque
import re
from minigrid.wrappers import FullyObsWrapper
from minigrid.manual_control import ManualControl

# ==========================================
# CONFIGURAZIONE VLM (Moondream2)
# ==========================================
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Caricamento del modello {model_id} in corso su {device}... (potrebbe richiedere qualche minuto)")
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

PROMPT = """Sei un sistema AI esperto in navigazione spaziale e deduzione logica.
Stai guidando un agente (il triangolo rosso) verso il suo obiettivo (il quadrato verde) in un GridWorld 2D. Non conosci le regole specifiche del gioco, devi dedurle dalla storia visiva.

REGOLE VISIVE:
- AGENTE: Triangolo rosso. La punta del triangolo indica la direzione frontale, osserva bene e non confonderti con gli altri vertici.
- GOAL: Quadrato verde.
- OSTACOLI FISSI (Muri): Blocchi grigi. Non possono essere attraversati. Se l'agente cerca di andarci contro, non si muoverà.
- ALTRI OGGETTI: Blocchi colorati (es. giallo). Possono essere aggirati o, se bloccano la strada, si può tentare di interagirci (TOGGLE o PICKUP).
- Spesso gli oggetti con lo stesso colore condividono qualche legame.

IL TUO COMPITO MENTALE OBBIGATORIO (Analisi):
1. Osserva l'immagine e deduci le regole dell'ambiente.
2. Identifica cosa deve fare l'agente per raggiungere il goal
3. che sequenza di azioni deve fare e con che oggetti deve interagire.

AZIONI DISPONIBILI:
0: LEFT (Ruota sul posto di 90 gradi a sinistra)
1: RIGHT (Ruota sul posto di 90 gradi a destra)
2: FORWARD (Avanza di un blocco nella direzione in cui guarda)
3: PICKUP (Raccogli)
4: DROP (Lascia)
5: TOGGLE (Interagisci)

FORMATO DI RISPOSTA OBBLIGATORIO:
Non aggiungere testo al di fuori di questi tag.

<ANALISIS>
(Max 200 parole). Scrivi la tua analisi. 1. osservazione, 2. identificazione obbiettivo, 3. sequenza azioni.
</ANALISIS>
"""


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

        user_text = (
            f"--- LOG DI SISTEMA ---\n"
            f"L'agente ha appena inviato l'azione numero: {self._last_action}\n"
            f"la reward dell'ambiente era: {self._last_env_reward}"
            f"----------------------\n\n"
            f"Esegui la tua analisi visiva confrontando i frame, valuta il risultato dell'azione {self._last_action} e fornisci le nuove metriche."
        )

        # Per Moondream2 concateniamo semplicemente il system prompt con il testo dell'utente
        question = f"{PROMPT}\n\n{user_text}"

        try:
            # Codifica dell'immagine per Moondream2
            enc_image = model.encode_image(image)
            
            # Generazione della risposta
            response = model.answer_question(enc_image, question, tokenizer)
            
        except Exception as e:
            response = f"Errore durante l'inferenza del modello: {e}"

        # --- DEBUG VISIVO: STAMPIAMO IL RAGIONAMENTO DEL MODELLO ---
        print("\n" + "=" * 50)
        print("🧠 RAGIONAMENTO VLM (Moondream2):")
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

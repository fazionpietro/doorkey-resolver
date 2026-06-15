import pandas as pd
import requests
import time
import sys
import os

# Configurazione
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "granite4.1:8b"  # o "llama3.1", "mistral", ecc.


def predict_rewards_with_llm(chunk_df, examples_text, debug=True):
    """Invia un blocco di dati al modello locale monitorando i parametri di debug."""

    chunk_to_predict = chunk_df.drop(columns=["reward_obtained"]).to_csv(index=False)

    prompt = f"""
You are an expert AI agent specializing in Reinforcement Learning and Dense Reward Shaping within a Doorkey (MiniGrid) environment.
Your task is to predict the 'reward_obtained' column based on state-action transitions.

You are a triangular agent in a 2D grid-world MiniGrid-DoorKey environment.

**OBJECTIVE**: "use the key to open the door and then get to the goal"
- Reach the green square (goal)

**CORE MECHANIC**:
1. Find the key in the grid.
2. Pick up the key with action `pickup` (action 3).
3. Go to the locked door.
4. Unlock the door with action `toggle` (action 5) — works ONLY if you have the key.
5. Reach the green goal.

**ACTIONS** (Discrete(7) space):
- 0: left (turn left)
- 1: right (turn right)
- 2: forward (move forward 1 cell)
- 3: pickup (pick up object — use for the key)
- 4: drop (not used)
- 5: toggle (activation — use to unlock door)
- 6: done (not used)

**OBSERVATION** (Dict):
- `direction`: 0-3 (current direction: south/north/west/east)
- `image`: 7x7x3 box (encoded local vision)
  - Each tile: (OBJECT_IDX, COLOR_IDX, STATE)
  - Door STATE: 0=open, 1=closed, 2=locked
- `mission`: string with objective

**REWARD SHAPING PRINCIPLES (Strict Range: [-1.0, 1.0])**:
Instead of relying on hardcoded values, evaluate state-action transitions based on these continuous progress mechanics:
- **Global Bounds**: The absolute maximum reward (+1.0) is strictly reserved for completing the final objective (reaching the green goal).
- **Dynamic Targeting**: Evaluate progress based on the *current active sub-goal* (Target Key -> Target Door -> Target Goal). 
- **Potential-Based Shaping**: Actions that decrease the spatial distance to the current active sub-goal should yield small positive rewards. Actions that increase distance or hit walls should yield 0 or slight negative penalties.
- **Milestone Spikes**: Completing a logical sub-goal (e.g., picking up the key, unlocking the door) should yield a significant positive reward proportional to its importance in the sequence.
- **Action Efficiency**: Penalize wasted actions (e.g., using `toggle` when not facing the door, or `pickup` when no key is present).

**EXAMPLES FOR CALIBRATION**:
Analyze the following dataset deeply. Deduce the mathematical baseline, scaling factors, and specific weights the environment uses for distances, penalties, and milestones. You must apply this exact same logic to your predictions.
{examples_text}

**TASK**:
For every tuple (obs_agent_x, obs_agent_y, obs_agent_dir, obs_key_pos, obs_door_open, obs_stage, action_taken) provided below, calculate the appropriate dense reward. Replace the '0' in 'reward_obtained' by perfectly mimicking the shaping logic deduced from the examples.

RETURN ONLY the completed CSV format, including the 'reward_obtained' column. Do NOT add any extra text, descriptions, explanations, or markdown formatting. Only raw CSV data.

**TO COMPLETE:**
{chunk_to_predict}"""

    if debug:
        print(
            f"  [DEBUG] Inviando prompt al modello (Lunghezza: {len(prompt)} caratteri)..."
        )

    start_time = time.time()

    try:
        # Aggiunto un timeout di 120 secondi per evitare che lo script si blocchi all'infinito
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0  # Temperatura a 0 per avere output deterministici e rigidi
                },
            },
            timeout=120,
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            raw_output = response.json().get("response", "")
            if debug:
                print(f"  [DEBUG] Risposta ricevuta in {elapsed_time:.2f} secondi.")
                # Stampa i primi 150 caratteri della risposta per controllare eventuali allucinazioni
                print(
                    f"  [DEBUG] Anteprima output grezzo: {raw_output[:150].replace(chr(10), ' | ')}..."
                )
            return raw_output
        else:
            print(
                f"  [ERRORE HTTP] Codice: {response.status_code} - Dettagli: {response.text}"
            )
            return None

    except requests.exceptions.Timeout:
        print(
            f"  [ERRORE TIMEOUT] Il modello ci ha messo più di 120 secondi a rispondere."
        )
        return None
    except Exception as e:
        print(
            f"  [ERRORE DI CONNESSIONE] Assicurati che Ollama sia in esecuzione. Dettagli: {e}"
        )
        return None


def process_file_agentically(
    subset_path, zero_path, output_path, chunk_size=30, debug=True
):
    print(f"[INFO] Caricamento file e preparazione del Few-Shot Prompting...")

    # Carica il subset denso ed estrae degli esempi
    try:
        subset_df = pd.read_csv(subset_path)
        examples_text = subset_df.sample(n=15, random_state=42).to_csv(index=False)
    except Exception as e:
        print(f"[ERRORE CRITICO] Impossibile leggere il subset di base: {e}")
        return

    print(
        f"[INFO] Inizio elaborazione a blocchi di {chunk_size} righe. Output incrementale su: {output_path}"
    )
    print("-" * 50)

    skip_rows = 0
    file_exists = os.path.exists(output_path)
    if file_exists:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    skip_rows = len(lines) - 1
                    print(
                        f"[INFO] Trovato file esistente con {skip_rows} righe già processate. Riprendo da lì..."
                    )
                elif len(lines) == 1:
                    print(
                        f"[INFO] Trovato file esistente ma solo con l'header. Ricomincio dall'inizio."
                    )
                    file_exists = False
                else:
                    file_exists = False
        except Exception as e:
            print(f"[AVVISO] Impossibile leggere {output_path}: {e}. Riparto da zero.")
            file_exists = False

    skip_range = range(1, skip_rows + 1) if skip_rows > 0 else None
    chunk_iter = pd.read_csv(zero_path, chunksize=chunk_size, skiprows=skip_range)

    first_chunk = not file_exists
    for i, chunk in enumerate(chunk_iter):
        row_start = skip_rows + i * chunk_size
        row_end = row_start + len(chunk) - 1
        print(f"\n--- Iterazione {i+1} | Righe da {row_start} a {row_end} ---")

        # Inferenza
        result_csv_text = predict_rewards_with_llm(chunk, examples_text, debug)

        if result_csv_text:
            # Pulizia dell'output da eventuali allucinazioni markdown (es. ```csv ... ```)
            result_csv_text = (
                result_csv_text.replace("```csv", "").replace("```", "").strip()
            )
            lines = result_csv_text.split("\n")

            # Gestione dell'header: lo manteniamo solo per il primo blocco
            if not first_chunk and len(lines) > 0 and "obs_agent_x" in lines[0]:
                lines = lines[1:]

            clean_text = "\n".join(lines) + "\n"

            # Salvataggio fisico sul disco
            mode = "w" if first_chunk else "a"
            try:
                with open(output_path, mode, encoding="utf-8") as f:
                    f.write(clean_text)
                    f.flush()  # FORZA IL SALVATAGGIO SUL DISCO AD OGNI ITERAZIONE
                print(f"  [INFO V] Blocco {i+1} salvato con successo sul disco.")
            except Exception as e:
                print(
                    f"  [ERRORE CRITICO] Impossibile scrivere sul file di output: {e}"
                )
                sys.exit(1)

            first_chunk = False
        else:
            print(f"\n[ERRORE CRITICO] Fallimento irreversibile sull'iterazione {i+1}.")
            print(
                f"[INFO] I blocchi precedenti (fino al {i}) sono salvati in sicurezza in {output_path}."
            )
            print("Interruzione dello script.")
            break


if __name__ == "__main__":
    process_file_agentically(
        "doorkey_sampled_subset.csv",
        "doorkey_sampled_zero_reward.csv",
        "doorkey_completed_llm.csv",
        chunk_size=30,  # Regola questo parametro in base alla memoria della tua GPU/RAM
        debug=True,  # Mantieni a True per vedere i log
    )

import sys
import yaml
from pathlib import Path

# Aggiungi la cartella 'src' al path per importare correttamente 'env'
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from env.rewardsystem import DoorKeyRewardSystem, RewardConfig


def debug_rewardsystem():
    print("=== INIZIALIZZAZIONE AMBIENTE E REWARD SYSTEM ===")
    config = RewardConfig()

    # Inizializziamo l'ambiente MiniGrid DoorKey (render_mode="human" per mostrare la finestra visiva)
    base_env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")

    # Applichiamo il nostro wrapper
    env = DoorKeyRewardSystem(base_env, config)

    # Fissiamo il seed per avere un layout deterministico e poter riprodurre i passi
    obs, info = env.reset()
    env.render()

    print("\n--- Valori post-reset() ---")
    print(f"Posizione Chiave (key_pos): {env.key_pos}")
    print(f"Posizione Porta (door_pos): {env.door_pos}")
    print(f"Posizione Obiettivo (goal_pos): {env.goal_pos}")
    print("\nDistanze di Riferimento per ogni Stage (stage_ref_distances):")
    for stage, dist in env.stage_ref_distances.items():
        print(f"  {stage}: {dist}")

    print(f"\nStage Corrente: {env.curr_stage}")
    print(f"Progresso Corrente: {env.curr_progress:.4f}")

    ACTION_NAMES = {
        0: "LEFT",
        1: "RIGHT",
        2: "FORWARD",
        3: "PICKUP",
        4: "DROP",
        5: "TOGGLE",
        6: "DONE",
    }

    print("\n\n=== CONTROLLO MANUALE ATTIVATO ===")
    print("Seleziona la finestra dell'ambiente e usa i seguenti tasti:")
    print("  - FRECCE DIREZIONALI: Su (avanti), Sinistra, Destra")
    print("  - TASTO E: Raccogli (Pickup)")
    print("  - TASTO R: Rilascia (Drop)")
    print("  - SPAZIO: Interagisci/Apri porta (Toggle)")
    print("  - TASTO ESC: Esci")

    import pygame

    running = True
    step_count = 0

    while running:
        # Gestiamo gli eventi di PyGame per ascoltare la tastiera
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    print("\nUscita manuale.")
                    break

                action = None
                if event.key == pygame.K_LEFT:
                    action = 0  # LEFT
                elif event.key == pygame.K_RIGHT:
                    action = 1  # RIGHT
                elif event.key == pygame.K_UP:
                    action = 2  # FORWARD
                elif event.key == pygame.K_e:
                    action = 3  # PICKUP
                elif event.key == pygame.K_r:
                    action = 4  # DROP
                elif event.key == pygame.K_SPACE:
                    action = 5  # TOGGLE

                if action is not None:
                    step_count += 1
                    action_name = ACTION_NAMES.get(action, f"UNKNOWN({action})")

                    print(f"\n--- STEP {step_count} | Azione: {action_name} ---")

                    # Eseguiamo l'azione
                    obs, total_reward, terminated, truncated, info = env.step(action)
                    env.render()

                    # Stampo il debug dettagliato richiesto
                    print(f"Eventi Estratti (curr_events):")
                    print(f"  has_key: {env.curr_events.has_key}")
                    print(f"  door_open: {env.curr_events.door_open}")
                    print(f"  goal_reached: {env.curr_events.goal_reached}")
                    print(f"Stage Dedotto (_infer_stage): {env.curr_stage}")

                    print(
                        f"Progresso Stage Calcolato (_compute_stage_progress): {env.curr_progress:.4f}"
                    )
                    print(
                        f"Variazione Progresso (_compute_progress_shaping): da {env.prev_progress:.4f} a {env.curr_progress:.4f}"
                    )

                    milestones = info.get("milestones", [])
                    regressions = info.get("regressions", [])
                    print(
                        f"Milestones Individuate (_detect_milestones): {milestones if milestones else 'Nessuna'}"
                    )
                    print(
                        f"Regressioni Individuate (_detect_regressions): {regressions if regressions else 'Nessuna'}"
                    )

                    rb = info.get("reward_breakdown", {})
                    print(f"Scomposizione Ricompensa (RewardBreakdown):")
                    print(f"  + env_reward:          {rb.get('env_reward', 0):.4f}")
                    print(f"  + stage_bonus:         {rb.get('stage_bonus', 0):.4f}")
                    print(
                        f"  + progress_shaping:    {rb.get('progress_shaping', 0):.4f}"
                    )
                    print(
                        f"  - regression_penalty:  {rb.get('regression_penalty', 0):.4f}"
                    )
                    print(f"  - time_penalty:        {rb.get('time_penalty', 0):.4f}")
                    print(
                        f"  = TOTAL REWARD ALL'AGENTE:  {rb.get('total', 0):.4f} (ritornato: {total_reward:.4f})"
                    )

                    if terminated or truncated:
                        print(
                            f"\n[!] Episodio terminato (Terminated: {terminated}, Truncated: {truncated})"
                        )
                        print(
                            "Faccio il reset dell'ambiente per farti continuare se vuoi...\n"
                        )
                        obs, info = env.reset(seed=42)
                        env.render()


if __name__ == "__main__":
    debug_rewardsystem()

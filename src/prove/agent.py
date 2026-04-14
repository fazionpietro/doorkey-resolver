from re import error
import gymnasium as gym
import minigrid  # Necessario per registrare gli environment MiniGrid
import numpy as np
import random
from collections import defaultdict
import time
from minigrid.wrappers import FullyObsWrapper
import wandb

# ==========================================
# TODO 1: Definisci gli Iperparametri
# ==========================================
# Imposta i valori per:
# - ALPHA (learning rate)
# - GAMMA (discount factor)
# - EPSILON_START, EPSILON_MIN, EPSILON_DECAY (per l'esplorazione)
# - NUM_EPISODES
ENV_ID = "MiniGrid-DoorKey-6x6-v0"
ALPHA = 0.0001
GAMMA = 0.9999
EPS_START = 1.0
EPS_END = 0.10
EPS_DECAY = 0.99995
N_EP = 15_000


def extract_state(obs):
    """
    MiniGrid restituisce un dizionario. Il Q-Learning richiede stati discreti.

    TODO 2: Estrai le informazioni rilevanti da `obs` e convertile in
    un formato immutabile (es. una tupla) che rappresenti univocamente la
    situazione dell'agente.
    """
    image = obs["image"]
    object_ids = tuple(image[:, :, 0].flatten())
    object_states = tuple(image[:, :, 2].flatten())

    return (object_ids, object_states)


def choose_action(state, q_table, epsilon, action_space):
    """
    TODO 3: Implementa la politica Epsilon-Greedy.

    - Con probabilità `epsilon`, restituisci un'azione casuale (Esplorazione).
    - Altrimenti, restituisci l'azione con il Q-value massimo per lo `state`
      corrente (Sfruttamento).
    """
    rand = random.uniform(0, 1)
    if rand < epsilon:
        return action_space.sample()
    else:
        q_values = [q_table[(state, a)] for a in range(action_space.n)]

        best_action = np.argmax(q_values)
        return best_action


def update_q_table(
    q_table, state, action, reward, next_state, alpha, gamma, action_space
):
    """
    TODO 4: Aggiorna il valore Q per la coppia (stato, azione).

    Usa la formula di Bellman per calcolare il nuovo Q-value e
    salvalo nella q_table.
    """

    current_q = q_table[(state, action)]
    next_q_values = [q_table[(next_state, a)] for a in range(action_space.n)]
    max_next_q = max(next_q_values)

    target = reward + gamma * max_next_q

    error = target - current_q

    q_table[(state, action)] = current_q + alpha * error


def train():
    run = wandb.init(
        project="minigrid-qlearning",
        name=f"DoorKey-6x6-qlearning",
        config={
            "env_id": ENV_ID,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay": EPS_DECAY,
            "num_episodes": N_EP,
        },
    )
    # Creiamo un ambiente piccolo per iniziare
    env = gym.make(ENV_ID)
    env = FullyObsWrapper(env)
    # Inizializziamo la Q-table.
    # Un defaultdict che restituisce 0.0 per stati mai visti è molto comodo qui.
    q_table = defaultdict(lambda: 0.0)
    epsilon = EPS_START

    for episode in range(N_EP):
        obs, info = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            # 1. Scegli l'azione
            action = choose_action(state, q_table, epsilon, env.action_space)

            # 2. Esegui l'azione nell'ambiente
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = extract_state(next_obs)

            # 3. Aggiorna la Q-Table
            update_q_table(
                q_table,
                state,
                action,
                reward,
                next_state,
                ALPHA,
                GAMMA,
                env.action_space,
            )
            # 4. Passa al nuovo stato
            state = next_state
            episode_reward += float(reward)

        # ==========================================
        # TODO 5: Decadimento di Epsilon
        # ==========================================
        # Riduci leggermente il valore di epsilon alla fine di ogni episodio
        # per passare gradualmente dall'esplorazione allo sfruttamento.
        epsilon = max(0.05, epsilon * EPS_DECAY)
        wandb.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "epsilon": epsilon,
                "q_table_size": len(q_table),
            }
        )

        if episode % 200 == 0:
            print(
                f"Episodio: {episode}, Epsilon: {epsilon:.3f}, Reward: {episode_reward:.3f}"
            )
    env.close()
    wandb.finish()
    return q_table


def test_agent(q_table):
    """
    Testa l'agente addestrato facendolo agire nell'ambiente visibile.
    """
    print("\n--- INIZIO FASE DI TEST ---")
    env = gym.make(ENV_ID, render_mode="human")
    env = FullyObsWrapper(env)

    obs, info = env.reset()
    state = extract_state(obs)

    done = False
    truncated = False
    reward = 0.0
    while not (done or truncated):
        # Sfruttamento totale, niente esplorazione
        action = choose_action(
            state, q_table, epsilon=0.0, action_space=env.action_space
        )

        # L'agente esegue l'azione
        obs, reward, done, truncated, info = env.step(action)
        state = extract_state(obs)

        # Pausa per vedere i movimenti a schermo
        time.sleep(0.2)

    if float(reward) > 0:
        print("VITTORIA! L'agente ha aperto la porta e raggiunto l'obiettivo!")
    else:
        print("FALLIMENTO. L'agente si è perso o il tempo è scaduto.")
    env.close()


if __name__ == "__main__":
    print("Avvio addestramento Q-Learning su DoorKey...")
    trained_q_table = train()
    print("Addestramento completato!")
    test_agent(trained_q_table)
    test_agent(trained_q_table)
    test_agent(trained_q_table)
    test_agent(trained_q_table)
    test_agent(trained_q_table)

import gymnasium as gym
import minigrid
import numpy as np
import random
from collections import defaultdict
import time
from minigrid.wrappers import FullyObsWrapper
import wandb

ENV_ID = "MiniGrid-DoorKey-8x8-v0"
ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99995
N_EP = 17_000
N_EP_SWEEP = 12_000
WARMUP_FRAC = 0.10
DECAY_FRAC = 0.75

sweep_config = {
    "method": "bayes",
    "metric": {"name": "success_rate_100", "goal": "maximize"},
    # early_terminate: ferma i run peggiori dopo min_iter step
    # eta=2 -> elimina il 50% peggiore ad ogni controllo
    # Richiede gestione dell'eccezione in wandb.log (vedi run_loop)
    "early_terminate": {"type": "hyperband", "min_iter": 1000, "eta": 2},
    "parameters": {
        "alpha": {"distribution": "log_uniform_values", "min": 0.001, "max": 0.5},
        "gamma": {"values": [0.95, 0.99, 0.999]},
        "eps_decay": {"values": [0.9999, 0.99995, 0.99999]},
        "eps_end": {"values": [0.05, 0.10]},
        "warmup_frac": {"values": [0.05, 0.10, 0.20]},
        "decay_frac": {"values": [0.60, 0.75, 0.85]},
    },
}


def safe_mean(lst):
    """Media sicura: restituisce 0.0 su lista vuota."""
    return float(np.mean(lst)) if lst else 0.0


def extract_state(obs):
    image = obs["image"]
    assert image.ndim == 3 and image.shape[2] >= 3, (
        f"Formato osservazione inatteso: shape={image.shape}. "
        "Hai applicato FullyObsWrapper?"
    )
    return (tuple(image[:, :, 0].flatten()), tuple(image[:, :, 2].flatten()))


def choose_action(state, q_table, epsilon, action_space):
    if random.uniform(0, 1) < epsilon:
        return action_space.sample()
    return int(np.argmax([q_table[(state, a)] for a in range(action_space.n)]))


def update_q_table(
    q_table, state, action, reward, next_state, alpha, gamma, action_space
):
    current_q = q_table[(state, action)]
    max_next_q = max(q_table[(next_state, a)] for a in range(action_space.n))
    q_table[(state, action)] = current_q + alpha * (
        reward + gamma * max_next_q - current_q
    )


def decrease_epsilon(
    episode,
    n_episodes,
    eps_start,
    eps_end,
    warmup_frac=WARMUP_FRAC,
    decay_frac=DECAY_FRAC,
):
    warmup_ep = int(n_episodes * warmup_frac)
    decay_ep = int(n_episodes * decay_frac)
    if episode < warmup_ep:
        return eps_start
    elif episode < warmup_ep + decay_ep:
        progress = (episode - warmup_ep) / decay_ep
        return eps_start - progress * (eps_start - eps_end)
    else:
        return eps_end


def run_loop(cfg, n_episodes):
    env = gym.make(cfg.env_id)
    env = FullyObsWrapper(env)

    q_table = defaultdict(lambda: 0.0)
    epsilon = cfg.eps_start
    success_history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        ep_reward = 0.0
        step_count = 0

        while not (done or truncated):
            action = choose_action(state, q_table, epsilon, env.action_space)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = extract_state(next_obs)
            update_q_table(
                q_table,
                state,
                action,
                reward,
                next_state,
                cfg.alpha,
                cfg.gamma,
                env.action_space,
            )
            state = next_state
            ep_reward += float(reward)
            step_count += 1

        epsilon = decrease_epsilon(
            episode,
            n_episodes,
            cfg.eps_start,
            cfg.eps_end,
            warmup_frac=getattr(cfg, "warmup_frac", WARMUP_FRAC),
            decay_frac=getattr(cfg, "decay_frac", DECAY_FRAC),
        )
        success = 1.0 if ep_reward > 0 else 0.0
        success_history.append(success)
        success_rate = safe_mean(success_history[-100:])

        log_data = {
            "episode": episode,
            "episode_reward": ep_reward,
            "episode_length": step_count,
            "success": success,
            "success_rate_100": success_rate,
            "epsilon": epsilon,
            # numero di coppie (stato, azione) visitate
            "q_pairs_visited": len(q_table),
            # numero di stati unici visitati (indipendente dalle azioni)
            "q_states_unique": len(set(k[0] for k in q_table.keys())),
        }

        # Metriche pesanti ogni 200 episodi o all'ultimo -- usa campionamento
        if (episode % 200 == 0 or episode == n_episodes - 1) and len(q_table) > 0:
            sample = random.sample(list(q_table.values()), min(1000, len(q_table)))
            log_data["mean_q_value"] = float(np.mean(sample))
            log_data["max_q_value"] = float(np.max(sample))

            print(
                f"[Ep {episode:>5}/{n_episodes}] "
                f"reward={ep_reward:.3f} | "
                f"eps={epsilon:.3f} | "
                f"success_rate={success_rate:.2f} | "
                f"q_states={log_data['q_states_unique']}"
            )

        # step=episode garantisce che l'asse X su W&B mostri gli episodi
        try:
            wandb.log(log_data, step=episode)
        except Exception:
            # early_terminate di Hyperband ha chiuso la connessione:
            # usciamo dal loop pulitamente senza crashare
            print(f"[Ep {episode}] Run terminato da Hyperband.")
            break

    env.close()
    return q_table


def train():
    """Chiamata dallo sweep. Usa N_EP_SWEEP episodi."""
    run = wandb.init(
        project="minigrid-qlearning",
        config={
            "env_id": ENV_ID,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay": EPS_DECAY,
            "n_episodes": N_EP_SWEEP,
        },
    )
    run_loop(run.config, run.config.n_episodes)
    wandb.finish()


def train_final(alpha, gamma, eps_decay, eps_end):
    """Training completo con i migliori iperparametri trovati dallo sweep."""
    run = wandb.init(
        project="minigrid-qlearning",
        name="final-training",
        config={
            "env_id": ENV_ID,
            "alpha": alpha,
            "gamma": gamma,
            "eps_start": EPS_START,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "n_episodes": N_EP,
            "run_type": "final",
        },
    )
    q_table = run_loop(run.config, run.config.n_episodes)
    wandb.finish()
    return q_table


def test_agent(q_table, n_runs=5):
    print(f"\\n--- FASE DI TEST ({n_runs} run) ---")
    env = gym.make(ENV_ID, render_mode="human")
    env = FullyObsWrapper(env)
    wins = 0

    for i in range(n_runs):
        obs, _ = env.reset()
        state = extract_state(obs)
        done = False
        truncated = False
        reward = 0.0

        while not (done or truncated):
            action = choose_action(
                state, q_table, epsilon=0.0, action_space=env.action_space
            )
            obs, reward, done, truncated, _ = env.step(action)
            state = extract_state(obs)
            time.sleep(0.1)

        if float(reward) > 0:
            wins += 1
            print(f"  Run {i+1}: VITTORIA ✓")
        else:
            print(f"  Run {i+1}: fallimento ✗")

    print(f"\\nRisultato: {wins}/{n_runs} vittorie")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniGrid Q-Learning + W&B")
    parser.add_argument("--mode", choices=["sweep", "train", "test"], default="train")
    parser.add_argument("--sweep_count", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--eps_decay", type=float, default=EPS_DECAY)
    parser.add_argument("--eps_end", type=float, default=EPS_END)
    args = parser.parse_args()

    if args.mode == "sweep":
        print(f"=== Avvio Sweep ({args.sweep_count} run x {N_EP_SWEEP} episodi) ===")
        sweep_id = wandb.sweep(sweep_config, project="minigrid-qlearning")
        wandb.agent(sweep_id, function=train, count=args.sweep_count)
        print("\\nSweep completato! Controlla wandb.ai per i migliori iperparametri.")
        print("Poi esegui: python script.py --mode train --alpha X --gamma Y ...")

    elif args.mode == "train":
        print(
            f"=== Training finale ===\\n"
            f"alpha={args.alpha}, gamma={args.gamma}, "
            f"eps_decay={args.eps_decay}, eps_end={args.eps_end}"
        )
        q_table = train_final(
            alpha=args.alpha,
            gamma=args.gamma,
            eps_decay=args.eps_decay,
            eps_end=args.eps_end,
        )
        print("Training completato!")
        test_agent(q_table, n_runs=5)

    elif args.mode == "test":
        print("Esegui prima --mode train per addestrare e testare in sequenza.")

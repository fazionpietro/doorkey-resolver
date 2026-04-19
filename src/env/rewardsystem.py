from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from inspect import currentframe
from typing import Any, cast

from . import doorkey_events as doorev
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from collections import deque


# ─────────────────────────────────────────────
# Enum che rappresenta le fasi sequenziali del task DoorKey.
# Il progresso va da NO_KEY (inizio) fino a GOAL_REACHED (fine).
# ─────────────────────────────────────────────
class Stage(Enum):
    NO_KEY = "no_key"
    HAS_KEY = "has_key"
    DOOR_OPEN = "door_open"
    GOAL_REACHED = "goal_reached"


# ─────────────────────────────────────────────
# Snapshot degli eventi booleani rilevanti in un dato timestep.
# Viene usato per rilevare milestone e regressioni tra due step consecutivi.
# ─────────────────────────────────────────────
@dataclass
class EventSnapshot:
    has_key: bool
    door_open: bool
    goal_reached: bool


# ─────────────────────────────────────────────
# Iperparametri del reward shaping. Possono essere modificati
# senza toccare la logica del wrapper.
# ────────────────────────────────────────────
@dataclass
class RewardConfig:
    key_bonus: float = 0.2
    door_bonus: float = 0.2
    goal_bonus: float = 0.4
    regression_penalty: float = -0.3
    time_penalty: float = -0.01
    shaping_scale: float = 0.5
    gamma: float = 0.99


# ─────────────────────────────────────────────
# Scomposizione del reward totale nelle sue componenti.
# Utile per il logging e il debugging.
# ─────────────────────────────────────────────
@dataclass
class RewardBreakdown:
    env_reward: float = 0.0
    stage_bonus: float = 0.0
    progress_shaping: float = 0.0
    regression_penalty: float = 0.0
    time_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.env_reward
            + self.stage_bonus
            + self.progress_shaping
            + self.regression_penalty
            + self.time_penalty
        )


# ─────────────────────────────────────────────
# Wrapper principale che sovrappone il reward shaping all'ambiente MiniGrid DoorKey.
# Implementa potential-based shaping, bonus milestone e penalità regressione.
# ─────────────────────────────────────────────
class DoorKeyRewardSystem(gym.Wrapper):
    def __init__(self, env: gym.Env, config: RewardConfig):
        super().__init__(env)
        self.config = config

        self.prev_events: EventSnapshot | None = None
        self.curr_events: EventSnapshot | None = None

        self.prev_stage: Stage | None = None
        self.curr_stage: Stage | None = None

        self.prev_progress: float = 0.0
        self.curr_progress: float = 0.0

        self.key_pos: tuple[int, int] | None = None
        self.door_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None

        self.stage_ref_distances: dict[Stage, int] = {}

        self.completed_milestones: set[str] = set()

    def reset(self, **kwargs):
        """
        Reinizializza il wrapper all'inizio di un nuovo episodio.
        Rileva le posizioni degli oggetti e calcola le distanze BFS di riferimento
        per normalizzare il progresso in ciascuna fase.
        """
        obs, info = self.env.reset(**kwargs)

        self._reset_tracker()

        base_env = self._get_base_env()
        agent_start = tuple(base_env.agent_pos)

        self.key_pos = self._find_stage_goal_position("key")
        self.door_pos = self._find_stage_goal_position("door")
        self.goal_pos = self._find_stage_goal_position("goal")

        # Calcola le distanze BFS ottimali tra le fasi successive.
        # Questi valori sono usati come riferimento per normalizzare il progresso.
        self.stage_ref_distances = {
            Stage.NO_KEY: max(1, self._bfs_distance(agent_start, self.key_pos)),
            Stage.HAS_KEY: max(1, self._bfs_distance(self.key_pos, self.door_pos)),
            Stage.DOOR_OPEN: max(
                1,
                self._bfs_distance(
                    self.door_pos,
                    self.goal_pos,
                    ignore_closed_door=True,
                ),
            ),
            Stage.GOAL_REACHED: 1,
        }

        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)

        self.curr_progress = self._compute_stage_progress(self.curr_stage)
        self.prev_progress = self.curr_progress

        return obs, info

    def step(self, action):
        """
        Esegue un passo nell'ambiente e calcola il reward composto.
        Il reward totale è la somma di: reward ambientale, bonus milestone,
        potential-based shaping, penalità regressione e penalità temporale.
        """

        if self.curr_events is None or self.curr_stage is None:
            raise RuntimeError(
                "Wrapper state not initialized. Call reset() before step()."
            )

        # Aggiorna lo stato "precedente" prima di eseguire il passo
        self.prev_events = self.curr_events
        self.prev_stage = self.curr_stage
        self.prev_progress = self.curr_progress

        # Esegui l'azione nell'ambiente sottostante
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Aggiorna lo stato "corrente" dopo il passo
        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)
        self.curr_progress = self._compute_stage_progress(self.curr_stage)

        # Rileva eventi positivi (milestone) e negativi (regressioni)
        milestones = self._detect_milestones(self.prev_events, self.curr_events)
        regressions = self._detect_regressions(
            self.prev_events,
            self.curr_events,
            self.prev_stage,
            self.curr_stage,
        )

        # step_time_penalty = -0.002 if action in (0, 1) else self.config.time_penalty
        step_time_penalty = self.config.time_penalty

        reward_parts = RewardBreakdown(
            env_reward=float(env_reward),
            stage_bonus=self._compute_stage_bonus(milestones),
            progress_shaping=self._compute_progress_shaping(
                self.prev_stage, self.curr_stage, self.prev_progress, self.curr_progress
            ),
            regression_penalty=self._compute_regression_penalty(regressions),
            time_penalty=step_time_penalty,
        )
        info = self._augment_info(info, reward_parts, milestones, regressions)

        return obs, reward_parts.total, terminated, truncated, info

    def _reset_tracker(self) -> None:
        """Azzera tutto lo stato interno del wrapper (chiamato a ogni reset)."""
        self.prev_events = None
        self.curr_events = None
        self.prev_stage = None
        self.curr_stage = None
        self.prev_progress = 0.0
        self.curr_progress = 0.0

        self.key_pos = None
        self.door_pos = None
        self.goal_pos = None
        self.stage_ref_distances = {}

        self.completed_milestones.clear()

    def _get_base_env(self) -> MiniGridEnv:
        """Restituisce l'ambiente MiniGrid sottostante (senza wrapper)."""
        return cast(MiniGridEnv, self.env.unwrapped)

    def _extract_events(self) -> EventSnapshot:
        """Legge lo stato corrente dell'ambiente e crea uno snapshot degli eventi."""
        return EventSnapshot(
            has_key=doorev.has_key(self),
            door_open=doorev.door_is_open(self),
            goal_reached=doorev.goal_reached(self),
        )

    def _infer_stage(self, events: EventSnapshot) -> Stage:
        """
        Deduce la fase corrente dagli eventi booleani.
        L'ordine di priorità è: goal > porta aperta > chiave > nessuno.
        """
        if events.goal_reached:
            return Stage.GOAL_REACHED
        elif events.door_open:
            return Stage.DOOR_OPEN
        elif events.has_key:
            return Stage.HAS_KEY
        return Stage.NO_KEY

    def _detect_milestones(
        self,
        prev_events: EventSnapshot,
        curr_events: EventSnapshot,
    ) -> set[str]:
        """
        Rileva le transizioni positive tra due snapshot consecutivi.
        Una milestone viene registrata solo al momento della sua prima occorrenza.
        """
        milestone: set[str] = set()

        if not prev_events.has_key and curr_events.has_key:
            milestone.add("picked_key")

        if not prev_events.door_open and curr_events.door_open:
            milestone.add("opened_door")

        if not prev_events.goal_reached and curr_events.goal_reached:
            milestone.add("reached_goal")

        return milestone

    def _detect_regressions(
        self,
        prev_events: EventSnapshot,
        curr_events: EventSnapshot,
        prev_stage: Stage,
        curr_stage: Stage,
    ) -> set[str]:
        """
        Rileva le transizioni negative tra due snapshot consecutivi.
        Le regressioni indicano che l'agente ha perso un progresso già acquisito.
        """
        regressions: set[str] = set()

        if prev_events.has_key and not curr_events.has_key:
            regressions.add("lost_key")

        if prev_events.door_open and not curr_events.door_open:
            regressions.add("closed_door")
        return regressions

    def _compute_stage_progress(self, stage: Stage) -> float:
        """
        Calcola il progresso normalizzato [0, 1] dell'agente nella fase corrente,
        usando la distanza BFS rispetto al target della fase e la distanza di riferimento.
        Un valore pari a 1.0 indica che il target è raggiunto.
        """
        base_env = self._get_base_env()
        agent_pos = tuple(base_env.agent_pos)
        if stage == Stage.GOAL_REACHED:
            return 1.0

        if self.key_pos is None or self.door_pos is None or self.goal_pos is None:
            raise RuntimeError("Stage targets not initialized. Call reset() first.")

        if stage == Stage.NO_KEY:
            target = self.key_pos
            dist = self._bfs_distance(agent_pos, target)

        elif stage == Stage.HAS_KEY:
            target = self.door_pos
            dist = self._bfs_distance(agent_pos, target)

        elif stage == Stage.DOOR_OPEN:
            target = self.goal_pos
            dist = self._bfs_distance(agent_pos, target, ignore_closed_door=True)
        else:
            return 0.0

        ref = self.stage_ref_distances.get(stage, 1)

        if dist >= 10**9:
            return 0.0

        progress = 1.0 - (dist / ref)
        return max(0.0, min(1.0, progress))

    def _compute_progress_shaping(
        self,
        prev_stage: Stage | None,
        curr_stage: Stage | None,
        prev_progress: float,
        curr_progress: float,
    ) -> float:
        """
        Calcola il termine di potential-based reward shaping (Ng et al., 1999).
        Il potenziale globale combina l'indice della fase con il progresso BFS locale,
        garantendo che lo shaping sia teoretically safe (non altera la politica ottima).
        Formula: F = γ * Φ(s') - Φ(s)
        """
        if prev_stage is None or curr_stage is None:
            return 0.0

        # Calcola il potenziale globale unendo Stage + Progresso BFS
        prev_potential = self._compute_stage_potential(prev_stage, prev_progress)
        curr_potential = self._compute_stage_potential(curr_stage, curr_progress)

        # Delta basato sul potenziale globale (Ng et al., 1999)
        delta = self.config.gamma * curr_potential - prev_potential
        return self.config.shaping_scale * delta

    def _compute_stage_bonus(self, milestones: set[str]) -> float:
        """
        Assegna bonus una-tantum per ogni milestone completata nell'episodio.
        Usa `completed_milestones` per evitare di assegnare lo stesso bonus più volte.
        """
        bonus = 0.0
        if "picked_key" in milestones and "picked_key" not in self.completed_milestones:
            bonus += self.config.key_bonus
            self.completed_milestones.add("picked_key")

        if (
            "opened_door" in milestones
            and "opened_door" not in self.completed_milestones
        ):
            bonus += self.config.door_bonus
            self.completed_milestones.add("opened_door")

        if (
            "reached_goal" in milestones
            and "reached_goal" not in self.completed_milestones
        ):
            bonus += self.config.goal_bonus
            self.completed_milestones.add("reached_goal")

        return bonus

    def _compute_regression_penalty(self, regressions: set[str]) -> float:
        """
        Applica una penalità fissa se l'agente ha perso la chiave o richiuso la porta.
        La penalità è applicata una volta per step (non per singola regressione).
        """
        penality = 0.0
        if "lost_key" in regressions or "closed_door" in regressions:
            penality += self.config.regression_penalty

        return penality

    def _compose_reward(
        self,
        env_reward: float,
        prev_progress: float,
        curr_progress: float,
        milestones: set[str],
        regressions: set[str],
    ) -> RewardBreakdown:
        """
        Metodo ausiliario per comporre il RewardBreakdown completo.
        Attualmente non utilizzato nel flusso principale (la composizione avviene in step()),
        ma può essere utile per test o ricalcoli offline.
        """
        return RewardBreakdown(
            env_reward=env_reward,
            stage_bonus=self._compute_stage_bonus(milestones),
            progress_shaping=self._compute_progress_shaping(
                self.prev_stage, self.curr_stage, prev_progress, curr_progress
            ),
            regression_penalty=self._compute_regression_penalty(regressions),
            time_penalty=self.config.time_penalty,
        )

    def _augment_info(
        self,
        info: dict[str, Any],
        reward_parts: RewardBreakdown,
        milestones: set[str],
        regressions: set[str],
    ) -> dict[str, Any]:
        """
        Arricchisce il dizionario `info` restituito da step() con tutte le
        informazioni di diagnostica: fase, progresso, eventi, milestone,
        regressioni e scomposizione dettagliata del reward.
        """
        info = dict(info)

        info["stage"] = self.curr_stage.value if self.curr_stage is not None else None
        info["completion"] = self._completion_percentage()

        info["events"] = {
            "has_key": (
                self.curr_events.has_key if self.curr_events is not None else False
            ),
            "door_open": (
                self.curr_events.door_open if self.curr_events is not None else False
            ),
            "goal_reached": (
                self.curr_events.goal_reached if self.curr_events is not None else False
            ),
        }

        info["milestones"] = sorted(milestones)
        info["regressions"] = sorted(regressions)
        info["completed_milestones"] = sorted(self.completed_milestones)
        info["stage_progress"] = self.curr_progress

        info["reward_breakdown"] = {
            "env_reward": reward_parts.env_reward,
            "stage_bonus": reward_parts.stage_bonus,
            "progress_shaping": reward_parts.progress_shaping,
            "regression_penalty": reward_parts.regression_penalty,
            "time_penalty": reward_parts.time_penalty,
            "total": reward_parts.total,
        }

        return info

    def _find_stage_goal_position(self, goal) -> tuple[int, int]:
        """
        Cerca nella griglia MiniGrid la posizione della cella con il tipo specificato
        (es. "key", "door", "goal"). Lancia RuntimeError se non trovata.
        """
        base_env = self._get_base_env()
        grid = base_env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is not None and obj.type == goal:
                    return (x, y)
        raise RuntimeError(goal + " not found in the grid")

    def _bfs_distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        ignore_closed_door: bool = False,
    ) -> int:
        """
        Calcola la distanza minima in passi tra `start` e `goal` usando BFS sulla griglia.
        I muri bloccano sempre il passaggio. Le porte chiuse bloccano il passaggio
        a meno che `ignore_closed_door=True` (usato per la fase DOOR_OPEN).
        Restituisce 10**9 se il goal non è raggiungibile.
        """
        if start == goal:
            return 0

        base_env = self._get_base_env()
        grid = base_env.grid
        width, height = grid.width, grid.height

        q = deque([(start[0], start[1], 0)])
        visited = {start}

        while q:
            x, y, dist = q.popleft()

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                pos = (nx, ny)

                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                if pos in visited:
                    continue

                # Se la cella è proprio il goal, la considero raggiunta
                # anche se contiene una porta chiusa
                if pos == goal:
                    return dist + 1

                cell = grid.get(nx, ny)

                blocked = False
                if cell is not None:
                    if cell.type == "wall":
                        blocked = True
                    elif cell.type == "door":
                        is_open = bool(getattr(cell, "is_open", False))
                        if not is_open and not ignore_closed_door:
                            blocked = True

                if blocked:
                    continue

                visited.add(pos)
                q.append((nx, ny, dist + 1))

        return 10**9

    def _completion_percentage(self) -> float:
        """
        Calcola la percentuale di completamento globale del task [0, 1].
        Combina l'indice della fase corrente con il progresso BFS locale,
        suddividendo il task in 4 fasi di ugual peso.
        """
        if self.curr_stage is None:
            return 0.0

        stage_index = {
            Stage.NO_KEY: 0,
            Stage.HAS_KEY: 1,
            Stage.DOOR_OPEN: 2,
            Stage.GOAL_REACHED: 3,
        }[self.curr_stage]

        n_stages = 4
        return min(1.0, (stage_index + self.curr_progress) / n_stages)

    def _compute_stage_potential(self, stage: Stage, stage_progress: float) -> float:
        """
        Trasforma la coppia (fase, progresso locale) in un potenziale globale scalare.
        Il potenziale cresce monotonamente con l'avanzamento nel task:
        - Stage.NO_KEY   con progress=0 → potenziale = 0.0
        - Stage.GOAL_REACHED con progress=1 → potenziale = 4.0 (massimo)
        Usato dal potential-based shaping per calcolare il delta di reward.
        """
        stage_index = {
            Stage.NO_KEY: 0,
            Stage.HAS_KEY: 1,
            Stage.DOOR_OPEN: 2,
            Stage.GOAL_REACHED: 3,
        }[stage]

        return stage_index + stage_progress

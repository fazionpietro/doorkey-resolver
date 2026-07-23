from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast
from collections import deque

import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv

# Presumo che tu abbia questo modulo locale
from . import doorkey_events as doorev


# ─────────────────────────────────────────────
# Enum che rappresenta le fasi sequenziali del task DoorKey.
# ─────────────────────────────────────────────
class Stage(Enum):
    NO_KEY = "no_key"
    HAS_KEY = "has_key"
    DOOR_OPEN = "door_open"
    GOAL_REACHED = "goal_reached"


# ─────────────────────────────────────────────
# Snapshot degli eventi booleani rilevanti in un dato timestep.
# ─────────────────────────────────────────────
@dataclass
class EventSnapshot:
    has_key: bool
    door_open: bool
    goal_reached: bool


# ─────────────────────────────────────────────
# Iperparametri del reward shaping.
# ────────────────────────────────────────────
@dataclass
class RewardConfig:
    key_bonus: float = 0.5
    door_bonus: float = 0.5
    goal_bonus: float = 1.0
    regression_penalty: float = -0.7
    time_penalty: float = -0.001
    shaping_scale: float = 0.6
    gamma: float = 0.99


# ─────────────────────────────────────────────
# Scomposizione del reward totale nelle sue componenti.
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
# Wrapper principale V2
# ─────────────────────────────────────────────
class DoorKeyRewardSystem(gym.Wrapper):
    def __init__(self, env: gym.Env, config: RewardConfig):
        super().__init__(env)
        self.config = config

        self.prev_events: EventSnapshot | None = None
        self.curr_events: EventSnapshot | None = None

        self.prev_stage: Stage | None = None
        self.curr_stage: Stage | None = None

        self.prev_stage_potential: float = 0.0
        self.curr_stage_potential: float = 0.0

        self.key_pos: tuple[int, int] | None = None
        self.door_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None

        self.stage_ref_distances: dict[Stage, int] = {}
        self.completed_milestones: set[str] = set()

        # Le 3 tabelle precalcolate (progresso da 1.0 a 0.0)
        self.table_key: dict[tuple[int, int], float] = {}
        self.table_door: dict[tuple[int, int], float] = {}
        self.table_goal: dict[tuple[int, int], float] = {}

        self.max_dist_key: int = 1
        self.max_dist_door: int = 1
        self.max_dist_goal: int = 1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_tracker()

        base_env = self._get_base_env()
        agent_start = tuple(base_env.agent_pos)

        self.key_pos = self._find_stage_goal_position("key")
        self.door_pos = self._find_stage_goal_position("door")
        self.goal_pos = self._find_stage_goal_position("goal")

        # Calcolo tabelle progresso (progresso 1.0 = sull'oggetto, 0.0 = massimo allontanamento)
        self.table_key, self.max_dist_key = self._compute_table_and_max(self.key_pos)
        self.table_door, self.max_dist_door = self._compute_table_and_max(self.door_pos)
        self.table_goal, self.max_dist_goal = self._compute_table_and_max(
            self.goal_pos, ignore_closed_door=True
        )

        # Calcolo delle distanze di riferimento iniziali per le fasi usando le tabelle!
        d_start_to_key = self._get_distance(agent_start, Stage.NO_KEY)
        d_key_to_door = self._get_distance(self.key_pos, Stage.HAS_KEY)
        d_door_to_goal = self._get_distance(self.door_pos, Stage.DOOR_OPEN)

        self.stage_ref_distances = {
            Stage.NO_KEY: max(1, d_start_to_key),
            Stage.HAS_KEY: max(1, d_key_to_door),
            Stage.DOOR_OPEN: max(1, d_door_to_goal),
            Stage.GOAL_REACHED: 1,
        }

        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)
        self.curr_stage_potential = self._compute_stage_potential(self.curr_stage)
        self.prev_stage_potential = self.curr_stage_potential

        return obs, info

    def step(self, action):
        if self.curr_events is None or self.curr_stage is None:
            raise RuntimeError(
                "Wrapper state not initialized. Call reset() before step()."
            )

        self.prev_events = self.curr_events
        self.prev_stage = self.curr_stage
        self.prev_stage_potential = self.curr_stage_potential

        obs, env_reward, terminated, truncated, info = self.env.step(action)

        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)

        # Rileva eventi positivi e negativi
        milestones = self._detect_milestones(self.prev_events, self.curr_events)
        regressions = self._detect_regressions(
            self.prev_events, self.curr_events, self.prev_stage, self.curr_stage
        )

        if "lost_key" in regressions:
            try:
                self.key_pos = self._find_stage_goal_position("key")
                self.table_key, self.max_dist_key = self._compute_table_and_max(
                    self.key_pos
                )

                # Ricalcolo dinamico della ref_dist in modo che riparta pulito
                base_env = self._get_base_env()
                agent_pos = tuple(base_env.agent_pos)
                dist = float(self._get_distance(agent_pos, Stage.NO_KEY))
                if dist == 1.0 and tuple(base_env.front_pos) == self.key_pos:
                    dist = 0.5
                self.stage_ref_distances[Stage.NO_KEY] = max(1.0, dist)
            except RuntimeError:
                pass

        if self.prev_stage != Stage.DOOR_OPEN and self.curr_stage == Stage.DOOR_OPEN:
            if self.goal_pos is None:
                raise RuntimeError("goal_pos is None: reset() was not called.")
            self.table_goal, self.max_dist_goal = self._compute_table_and_max(
                self.goal_pos, ignore_closed_door=False
            )

        self.curr_stage_potential = self._compute_stage_potential(self.curr_stage)

        reward_parts = RewardBreakdown(
            env_reward=float(env_reward),
            stage_bonus=self._compute_stage_bonus(milestones),
            progress_shaping=self._compute_progress_shaping(
                self.prev_stage,
                self.curr_stage,
                self.prev_stage_potential,
                self.curr_stage_potential,
                terminated,
            ),
            regression_penalty=self._compute_regression_penalty(regressions),
            time_penalty=self.config.time_penalty,
        )

        info = self._augment_info(info, reward_parts, milestones, regressions)

        return obs, reward_parts.total, terminated, truncated, info

    def _reset_tracker(self) -> None:
        self.prev_events = None
        self.curr_events = None
        self.prev_stage = None
        self.curr_stage = None
        self.prev_stage_potential = 0.0
        self.curr_stage_potential = 0.0

        self.key_pos = None
        self.door_pos = None
        self.goal_pos = None
        self.stage_ref_distances = {}
        self.completed_milestones.clear()

        self.table_key = {}
        self.table_door = {}
        self.table_goal = {}
        self.max_dist_key = 1
        self.max_dist_door = 1
        self.max_dist_goal = 1

    def _get_base_env(self) -> MiniGridEnv:
        return cast(MiniGridEnv, self.env.unwrapped)

    def _extract_events(self) -> EventSnapshot:
        return EventSnapshot(
            has_key=doorev.has_key(self),
            door_open=doorev.door_is_open(self),
            goal_reached=doorev.goal_reached(self),
        )

    def _infer_stage(self, events: EventSnapshot) -> Stage:
        if events.goal_reached:
            return Stage.GOAL_REACHED
        elif events.door_open:
            return Stage.DOOR_OPEN
        elif events.has_key:
            return Stage.HAS_KEY
        return Stage.NO_KEY

    def _detect_milestones(
        self, prev_events: EventSnapshot, curr_events: EventSnapshot
    ) -> set[str]:
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
        regressions: set[str] = set()
        if prev_events.has_key and not curr_events.has_key:
            regressions.add("lost_key")
        if prev_events.door_open and not curr_events.door_open:
            regressions.add("closed_door")
        return regressions

    def _compute_table_and_max(
        self, goal: tuple[int, int], ignore_closed_door: bool = False
    ) -> tuple[dict[tuple[int, int], float], int]:
        """
        Calcola tramite BFS la distanza da 'goal' per tutte le celle.
        Restituisce la tabella con i progressi normalizzati tra 0.0 (lontano) e 1.0 (sul goal),
        insieme alla distanza massima, in modo da poter denormalizzare e riottenere le distanze.
        """
        base_env = self._get_base_env()
        grid = base_env.grid
        dist_map: dict[tuple[int, int], int] = {}
        q: deque = deque([(goal[0], goal[1], 0)])
        dist_map[goal] = 0

        while q:
            x, y, d = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                pos = (nx, ny)
                if pos in dist_map:
                    continue
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue

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

                dist_map[pos] = d + 1
                q.append((nx, ny, d + 1))

        if not dist_map:
            return {}, 1

        val_min = 0  # distance at goal is always 0
        val_max = max(dist_map.values())
        if val_max == 0:
            val_max = 1

        normalized: dict[tuple[int, int], float] = {}
        for pos, d in dist_map.items():
            norm_inverted = 1.0 - (d / val_max)
            normalized[pos] = round(norm_inverted, 3)

        return normalized, val_max

    def _get_distance(self, pos: tuple[int, int], stage: Stage) -> int:
        """
        De-normalizza il progresso estratto dalla tabella della fase per restituire la
        distanza reale calcolata col BFS originario, nascondendo il fatto che
        deriva da una tabella.
        """
        if stage == Stage.NO_KEY:
            table = self.table_key
            max_dist = self.max_dist_key
        elif stage == Stage.HAS_KEY:
            table = self.table_door
            max_dist = self.max_dist_door
        elif stage == Stage.DOOR_OPEN:
            table = self.table_goal
            max_dist = self.max_dist_goal
        else:
            return 0

        progress_val = table.get(pos, 0.0)
        return round((1.0 - progress_val) * max_dist)

    def _compute_stage_potential(self, stage: Stage) -> float:
        """
        Bilanciamento Step-Reward: calcola il potenziale usando il massimo assoluto
        di tutta la mappa (global_max_dist). In questo modo, OGNI singolo passo verso
        l'obiettivo darà la STESSA identica porzione di reward, indipendentemente
        da quanto sia lungo o corto lo stage corrente.
        """
        if stage == Stage.GOAL_REACHED:
            return 1.0

        base_env = self._get_base_env()
        agent_pos = tuple(base_env.agent_pos)

        current_dist = float(self._get_distance(agent_pos, stage))

        # Consapevolezza Direzionale: se adiacente, controlla se guarda l'oggetto
        if current_dist == 1.0:
            goal_pos = None
            if stage == Stage.NO_KEY:
                goal_pos = self.key_pos
            elif stage == Stage.HAS_KEY:
                goal_pos = self.door_pos
            elif stage == Stage.DOOR_OPEN:
                goal_pos = self.goal_pos

            if goal_pos and tuple(base_env.front_pos) == goal_pos:
                current_dist = 0.5

        global_max_dist = float(
            max(self.max_dist_key, self.max_dist_door, self.max_dist_goal)
        )
        if global_max_dist == 0:
            global_max_dist = 1.0

        return 1.0 - (current_dist / global_max_dist)

    def _compute_progress_shaping(
        self,
        prev_stage: Stage | None,
        curr_stage: Stage | None,
        prev_potential: float,
        curr_potential: float,
        terminated: bool,
    ) -> float:
        if prev_stage is None or curr_stage is None:
            return 0.0

        if terminated:
            curr_pot = 0.0
        else:
            curr_pot = curr_potential

        if prev_stage != curr_stage:
            return 0.0

        # Se il potenziale non cambia (es. rotazione sul posto),
        # restituiamo 0.0 per evitare il "gamma bleed".
        if abs(curr_pot - prev_potential) < 1e-6:
            return 0.0

        delta = self.config.gamma * curr_pot - prev_potential
        return delta * self.config.shaping_scale

    def _compute_stage_bonus(self, milestones: set[str]) -> float:
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
        penalty = 0.0
        if "lost_key" in regressions or "closed_door" in regressions:
            penalty += self.config.regression_penalty
        return penalty

    def _compute_global_progress(self) -> float:
        """
        Sfrutta le tabelle di progresso (de-normalizzando) per calcolare
        il progresso esatto lungo l'intero percorso ideale dell'episodio,
        usando le distanze massime (assolute) invece che la posizione iniziale.
        """
        if self.curr_stage is None:
            return 0.0
        if self.curr_stage == Stage.GOAL_REACHED:
            return 1.0

        base_env = self._get_base_env()
        agent_pos = tuple(base_env.agent_pos)

        # Usiamo le distanze massime possibili dell'ambiente
        D_k = self.max_dist_key
        D_d = self.max_dist_door
        D_g = self.max_dist_goal

        total_dist = D_k + D_d + D_g
        if total_dist == 0:
            return 1.0

        curr_dist = self._get_distance(agent_pos, self.curr_stage)

        if self.curr_stage == Stage.NO_KEY:
            remaining = curr_dist + D_d + D_g
        elif self.curr_stage == Stage.HAS_KEY:
            remaining = curr_dist + D_g
        elif self.curr_stage == Stage.DOOR_OPEN:
            remaining = curr_dist
        else:
            remaining = 0

        progress = 1.0 - (remaining / total_dist)
        return max(0.0, min(1.0, progress))

    def _augment_info(
        self,
        info: dict[str, Any],
        reward_parts: RewardBreakdown,
        milestones: set[str],
        regressions: set[str],
    ) -> dict[str, Any]:
        info = dict(info)
        info["stage"] = self.curr_stage.value if self.curr_stage is not None else None
        info["completion"] = self._compute_global_progress()
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
        info["stage_progress"] = self.curr_stage_potential

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
        base_env = self._get_base_env()
        grid = base_env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is not None and obj.type == goal:
                    return (x, y)
        raise RuntimeError(goal + " not found in the grid")

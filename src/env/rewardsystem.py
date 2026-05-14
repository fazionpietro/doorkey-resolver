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
    shaping_scale: float = 0.7
    gamma: float = 0.995


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
# Wrapper principale
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
        obs, info = self.env.reset(**kwargs)
        self._reset_tracker()

        base_env = self._get_base_env()
        agent_start = tuple(base_env.agent_pos)

        self.key_pos = self._find_stage_goal_position("key")
        self.door_pos = self._find_stage_goal_position("door")
        self.goal_pos = self._find_stage_goal_position("goal")

        self._dist_to_key = self._bfs_full_map(self.key_pos)
        self._dist_to_door = self._bfs_full_map(self.door_pos)
        self._dist_to_goal = self._bfs_full_map(self.goal_pos, ignore_closed_door=True)

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
        if self.curr_events is None or self.curr_stage is None:
            raise RuntimeError(
                "Wrapper state not initialized. Call reset() before step()."
            )

        self.prev_events = self.curr_events
        self.prev_stage = self.curr_stage
        self.prev_progress = self.curr_progress

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
                self._dist_to_key = self._bfs_full_map(self.key_pos)
            except RuntimeError:
                pass

        if self.prev_stage != Stage.DOOR_OPEN and self.curr_stage == Stage.DOOR_OPEN:
            if self.goal_pos is None:
                raise RuntimeError("goal_pos is None: reset() was not called.")
            self._dist_to_goal = self._bfs_full_map(
                self.goal_pos, ignore_closed_door=False
            )

        self.curr_progress = self._compute_stage_progress(self.curr_stage)

        reward_parts = RewardBreakdown(
            env_reward=float(env_reward),
            stage_bonus=self._compute_stage_bonus(milestones),
            progress_shaping=self._compute_progress_shaping(
                self.prev_stage,
                self.curr_stage,
                self.prev_progress,
                self.curr_progress,
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
        self.prev_progress = 0.0
        self.curr_progress = 0.0

        self.key_pos = None
        self.door_pos = None
        self.goal_pos = None
        self.stage_ref_distances = {}
        self.completed_milestones.clear()

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

    def _compute_stage_progress(self, stage: Stage) -> float:
        base_env = self._get_base_env()
        agent_pos = tuple(base_env.agent_pos)

        if stage == Stage.GOAL_REACHED:
            return 1.0
        elif stage == Stage.NO_KEY:
            return self._dist_to_key.get(agent_pos, 0.0)
        elif stage == Stage.HAS_KEY:
            return self._dist_to_door.get(agent_pos, 0.0)
        elif stage == Stage.DOOR_OPEN:
            return self._dist_to_goal.get(agent_pos, 0.0)
        return 0.0

    def _compute_stage_potential(self, stage: Stage, stage_progress: float) -> float:
        """
        [FIX CRITICO] Prevenzione del "Gamma Bleed" e disaccoppiamento fasi.
        Ora il potenziale resta SEMPRE confinato in [0, 1]. Non sommiamo più l'indice
        della fase, azzerando le penalità incontrollabili nelle fasi avanzate.
        """
        return stage_progress

    def _compute_progress_shaping(
        self,
        prev_stage: Stage | None,
        curr_stage: Stage | None,
        prev_progress: float,
        curr_progress: float,
        terminated: bool,
    ) -> float:
        """
        [FIX CRITICI] Risolto il Transition Drop e aggiunto il supporto a terminated.
        """
        if prev_stage is None or curr_stage is None:
            return 0.0

        # 1. Regola aurea del PBRS: se terminated, potenziale futuro DEVE essere 0.
        if terminated:
            curr_potential = 0.0
        else:
            curr_potential = self._compute_stage_potential(curr_stage, curr_progress)

        # 2. Spegni lo shaping durante il salto tra due fasi (Transition Drop).
        # Evita penalità massicce false quando il progresso crolla da 1.0 a 0.0.
        # L'agente verrà ricompensato esclusivamente dallo stage_bonus in questo step.
        if prev_stage != curr_stage:
            return 0.0

        prev_potential = self._compute_stage_potential(prev_stage, prev_progress)

        delta = self.config.gamma * curr_potential - prev_potential
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

    def _augment_info(
        self,
        info: dict[str, Any],
        reward_parts: RewardBreakdown,
        milestones: set[str],
        regressions: set[str],
    ) -> dict[str, Any]:
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

    def _bfs_full_map(
        self, goal: tuple[int, int], *, ignore_closed_door: bool = False
    ) -> dict[tuple, float]:
        base_env = self._get_base_env()
        grid = base_env.grid
        dist_map: dict[tuple, int] = {}
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
            return {}

        val_min = min(dist_map.values())
        val_max = max(dist_map.values())

        normalized: dict[tuple, float] = {}
        for pos, d in dist_map.items():
            if val_max > val_min:
                norm_inverted = 1.0 - (d - val_min) / (val_max - val_min)
            else:
                norm_inverted = 1.0
            normalized[pos] = round(norm_inverted, 3)

        return normalized

    def _completion_percentage(self) -> float:
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

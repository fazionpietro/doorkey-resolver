import gymnasium as gym
from typing import cast
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv


def has_key(env: gym.Env) -> bool:
    # controlla se l'agente trasporta una chiave
    base_env = cast(MiniGridEnv, env.unwrapped)
    return base_env.carrying is not None and base_env.carrying.type == "key"


def door_is_open(env: gym.Env) -> bool:
    # controlla lo stato della porta
    base_env = cast(MiniGridEnv, env.unwrapped)
    grid = base_env.grid
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == "door":
                return bool(getattr(obj, "is_open", False))
    return False


def goal_reached(env: gym.Env) -> bool:
    # controlla se l'agente è sul goal
    base_env = cast(MiniGridEnv, env.unwrapped)
    ax, ay = base_env.agent_pos
    grid = base_env.grid

    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == "goal":
                if ax == x and ay == y:
                    return True
    return False


def get_events(env: gym.Env) -> dict[str, bool]:
    return {
        "has_key": has_key(env),
        "door_open": door_is_open(env),
        "goal_reached": goal_reached(env),
    }

import gymnasium as gym

from .rewardsystem import DoorKeyRewardSystem, RewardConfig

DEFAULT_ENV_ID = "MiniGrid-DoorKey-5x5-v0"


def make_env(render_mode=None, reward_config=None):
    env = gym.make(DEFAULT_ENV_ID, render_mode=render_mode)

    if reward_config is None:
        reward_config = RewardConfig()

    env = DoorKeyRewardSystem(env, reward_config)
    return env

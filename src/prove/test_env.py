import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
env = FullyObsWrapper(gym.make("MiniGrid-DoorKey-8x8-v0"))
obs, _ = env.reset()
agent_in_image = False
for y in range(8):
    for x in range(8):
        if obs["image"][x, y, 0] == 10: # 10 is agent in minigrid
            agent_in_image = True
print("Agent in image:", agent_in_image)
print("Keys in obs:", obs.keys())
print("Direction:", obs["direction"])
import numpy as np
print("Image shape:", obs["image"].shape)

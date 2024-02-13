from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# from PakuPakuEnv import PakuPakuEnv
from PakuPakuEnv2 import PakuPakuEnv
from pathlib import Path
import os
import random
import numpy as np

import torch

SEED = 1337
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True





def test():
    model_name = f"ppo_reward_cap_vec_stack"
    env = PakuPakuEnv
    vec_env = VecFrameStack(VecNormalize(make_vec_env(env, n_envs=1, vec_env_cls=SubprocVecEnv)), n_stack=6, channels_order='last')
    model = PPO.load(f"trained_models/{model_name}.zip", vec_env, )
    obs = vec_env.reset()
    states = None
    deterministic = True
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        # input("Press enter to continue")
        
        actions, states = model.predict(
            obs,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        obs, rewards, dones, infos = vec_env.step(actions)
        
        episode_starts[0] = dones[0]
        print(infos[0]["score"], infos[0]["desc"], infos[0]["game_ended"], infos[0]['reward'])
        print(infos[0])
        
        if dones[0]:
            break

    vec_env.close()


if __name__ == "__main__":
    test()

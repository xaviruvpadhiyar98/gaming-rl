from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box

# from PakuPakuEnv import PakuPakuEnv
from PakuPakuEnv2 import PakuPakuEnv
from pathlib import Path
import os
import random
import torch as th
import torch.nn as nn


SEED = 1337
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


class EvalCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        if len(infos) != 1:
            sorted_infos = sorted(infos, key=lambda x: x['score'], reverse=True)
            best_info = sorted_infos[0]
        else:
            best_info = infos[0]

        if not best_info['game_ended']:
            return True
        for k, v in best_info.items():
            self.logger.record(f"xcommons/{k}", v)
        return True

    def _on_rollout_end(self) -> None:
        pass


    def _on_training_end(self) -> None:
        pass




class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for channel-last observation inputs, adapted to reorder inputs to channel-first.
    
    :param observation_space: (gym.Space) The observation space.
    :param features_dim: (int) Number of features extracted. This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Assuming channel-last format for the input, adjust the number of input channels accordingly.
        # observation_space.shape is expected to be (height, width, channels)
        n_input_channels = observation_space.shape[-1]  # This will fetch the number of channels

        # Define the CNN architecture
        self.cnn = nn.Sequential(
            # Adjust the first Conv2d layer to take `n_input_channels` as input
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the shape by doing one forward pass with a sample observation
        with th.no_grad():
            # Creating a sample observation based on the observation space's shape
            sample_obs = observation_space.sample().reshape(1, *observation_space.shape)  # Adding batch dimension
            # Reordering from channel-last to channel-first
            sample_obs = sample_obs.transpose(0, 3, 1, 2)  # From [batch, height, width, channels] to [batch, channels, height, width]
            n_flatten = self.cnn(th.as_tensor(sample_obs).float()).shape[1]

        # Define the final linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Reorder observations from channel-last to channel-first
        observations = observations.permute(0, 3, 1, 2)  # Assuming observations are in [batch, height, width, channels]
        # Pass the reordered observations through the CNN
        cnn_output = self.cnn(observations)
        # Pass the CNN's output through the linear layer
        return self.linear(cnn_output)




def train():
    model_name = f"ppo_reward_cap_vec_stack"
    TOTAL_TIMESTEPS = 200_000
    env = PakuPakuEnv
    vec_env = VecFrameStack(VecNormalize(make_vec_env(env, n_envs=4, vec_env_cls=SubprocVecEnv)), n_stack=4, channels_order='last')
    ENT_COEF = 0.06
    N_EPOCHS = 20
    N_STEPS = 2048
    BATCH_SIZE = 32
    GAMMA = 0.98
    CLIP_RANGE_VF = 0.2


    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = PPO.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="auto",
            ent_coef=ENT_COEF,
            n_epochs=N_EPOCHS,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            # clip_range_vf=CLIP_RANGE_VF,
            # normalize_advantage=False,
            verbose=2,
            # use_sde=True,
            # sde_sample_freq=4,
        )
    else:
        reset_num_timesteps = True
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=2,
            # buffer_size=100_000,
            n_steps=N_STEPS,
            device="auto",
            ent_coef=ENT_COEF,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            # clip_range_vf=CLIP_RANGE_VF,
            # normalize_advantage=False,
            tensorboard_log="tensorboard_log",
            # use_sde=True,
            # sde_sample_freq=4,
            policy_kwargs=dict(
                # net_arch=dict(pi=[1024, 2048, 1024], vf=[1024, 2048, 1024])
                normalize_images=False,
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128),
            ),
        )

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
            callback=EvalCallback(),
            tb_log_name=model_name,
        )
    except Exception as e:
        print(e)
        pass
    model.save("trained_models/" + model_name)
    vec_env.close()


if __name__ == "__main__":
    train()

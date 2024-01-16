from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from PakuPakuEnv import PakuPakuEnv
from pathlib import Path




class EvalCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:

        infos = self.locals["infos"]
        best_info = infos[0]
        for k, v in best_info.items():
            self.logger.record(f"xcommons/{k}", v)

    def _on_training_end(self) -> None:
        pass









def train():
    model_name = f"ppo"
    TOTAL_TIMESTEPS = 100_000
    env = PakuPakuEnv
    vec_env = make_vec_env(env)
    # ENT_COEF = 0.3
    N_EPOCHS = 20
    N_STEPS = 256

    if Path(f"trained_models/{model_name}.zip").exists():
        reset_num_timesteps = False
        model = PPO.load(
            f"trained_models/{model_name}.zip",
            vec_env,
            print_system_info=True,
            device="auto",
            # ent_coef=ENT_COEF,
            n_epochs=N_EPOCHS,
            n_steps=N_STEPS,
            verbose=0
        )
    else:
        reset_num_timesteps = True
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=0,
            n_steps=N_STEPS,
            device="auto",
            # ent_coef=ENT_COEF,
            n_epochs=N_EPOCHS,
            tensorboard_log="tensorboard_log",
            policy_kwargs = dict(
                net_arch=dict(pi=[1024, 2048, 1024], vf=[1024, 2048, 1024])
            )
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
        callback=EvalCallback(),
        tb_log_name=model_name,
    )
    model.save("trained_models/" + model_name)
    vec_env.close()


if __name__ == "__main__":
    train()
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
from PIL import Image
import json
from base64 import b64decode
from io import BytesIO
from time import sleep
from collections import Counter





class PakuPakuEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self):
        super(PakuPakuEnv, self).__init__()
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=255, shape=(50, 100, 1), dtype=np.uint8)
        self.is_browser_open = False
        options = Options()
        options.binary_location = (Path.home() / "Softwares/firefox/firefox").as_posix()
        options.add_argument("-headless")
        self.driver = Firefox(service=Service(executable_path=(Path.cwd() / "geckodriver").as_posix()), options=options)
        self.driver.get("http://localhost:4000/?pakupaku")
        ActionChains(self.driver).send_keys(Keys.ENTER).perform()





    def get_file_values(self):
        while not self.record1_file.exists():
            sleep(0.01)
        text = self.record1_file.read_text()
        if not text:
            sleep(0.01)
            text = self.record1_file.read_text()
        try:
            record1 = json.loads(text)
        except Exception as e:
            print(e)
            print(text, type(text))
        base64_img_data = record1["screenshot"].replace("data:image/png;base64,", "")
        bw_image = Image.open(BytesIO(b64decode(base64_img_data))).convert('L')
        # bw_image.save("imgs/1.png", format="png")
        observation = np.expand_dims(np.asarray(bw_image), axis=-1)
        score = record1["score"]
        game_ended = record1["gameEnded"]
        player_position = record1["playerPosition"]
        enemy_position = record1["enemyPosition"]
        self.record1_file.unlink(missing_ok=True)
        self.action1_file.unlink(missing_ok=True)
        return observation, score, game_ended, player_position, enemy_position


    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.driver.refresh()
        ActionChains(self.driver).send_keys(Keys.ENTER).perform()
        self.env_record = Path("env_record")
        self.record1_file = (self.env_record / "record1")
        self.action1_file = (self.env_record / "action1")
        observation, score, game_ended, player_position, enemy_position = self.get_file_values()
        self.previous_score = 0
        self.reward_tracker = 0
        self.action_tracker = ''
        self.previous_action = ''
        info = {
            "score": score,
            "game_ended": game_ended,
            "player_position": player_position,
            "enemy_position": enemy_position
        }
        return observation, info

    def step(self, action):
        terminated = False
        done = False

        action = str(action)
        self.action1_file.write_text(action)
        self.action_tracker += action
        
        observation, score, game_ended, player_position, enemy_position = self.get_file_values()
        reward = score - self.previous_score
        diff = abs(player_position - enemy_position)

        if reward == 0 and diff < 11:
            reward = -1
        elif reward == 0 and diff > 11:
            reward = 0.5
        else:
            reward = 1

        self.previous_score = score
        self.reward_tracker += reward
        info = {
            "reward": reward,
            "reward_tracker": self.reward_tracker,
            "score": score,
            "game_ended": game_ended,
            "player_position": player_position,
            "enemy_position": enemy_position
        }
        
        if game_ended:
            reward -= 10
            common_res = Counter(self.action_tracker).most_common(2)
            for c in common_res:
                info[c[0]] = c[1]
            print(info)
            terminated = True
            done = False
        
        reward = max(min(reward, 1), -1)
        return observation, reward, terminated, done, info
    
    def render(self):
        ...
        # if self.render_mode == "rgb_array":
        #     return self._render_frame()



    def close(self):
        self.driver.quit()
        self.record1_file.unlink(missing_ok=True)
        self.action1_file.unlink(missing_ok=True)


if __name__=="__main__":
    from time import sleep
    from random import SystemRandom
    env = PakuPakuEnv()
    env.reset()
    for r in range(100):
        action = SystemRandom().choice([0,1])
        observation, reward, terminated, done, info = env.step(action)
        if done or terminated:
            break
    env.close()
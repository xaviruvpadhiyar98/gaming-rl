import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from pathlib import Path
from PIL import Image
from base64 import b64decode
from io import BytesIO
from collections import Counter
from time import sleep
import logging
from torch.utils.tensorboard import SummaryWriter
import json
from typing import Any, List
from ast import literal_eval
from datetime import datetime

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s %(threadName)-11s [%(filename)s:%(lineno)d %(funcName)5s()] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PakuPakuEnv(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        super(PakuPakuEnv, self).__init__()
        self.action_space = Discrete(2)
        # self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float16)
        # self.observation_space = Box(
        #     low=0, high=255, shape=(50, 100, 4), dtype=np.uint8
        # )
        self.observation_space = Box(
            low=0, high=255, shape=(50, 100, 4), dtype=np.int32
        )

        # Selenium
        options = ChromeOptions()
        options.binary_location = (
            Path.home() / "Softwares/thorium-browser_117.0.5938.157_amd64/thorium"
        ).as_posix()
        # options.add_argument("--auto-open-devtools-for-tabs")
        # options.add_argument("--disable-gpu")
        # options.add_argument('--disable-dev-shm-usage')
        # options.add_argument("--headless")
        options.add_argument("--disable-infobars")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=250,250")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.set_capability("goog:loggingPrefs", {'browser': "ALL"})

        self.driver = Chrome(service=ChromeService(), options=options)
        self.driver.get("http://localhost:4000/?pakupaku")
        ActionChains(self.driver).send_keys(Keys.ENTER).perform()
        self.driver.get_log('browser')




    def get_obs(self, record):
        b64_img = record['screenshot'].replace("data:image/png;base64,", "")
        # logger.info(b64_img)
        img_bytes = (b64decode(b64_img))
        # current_datetime = str(int(datetime.utcnow().timestamp()*1000000))
        # (Path("imgs") / f"{current_datetime}.png").write_bytes(img_bytes)
        img = Image.open(BytesIO(img_bytes))
        return np.asarray(img)

    def get_browser_log(self, reset=False):
        if reset:
            sleep(0.01)
            # print("reseted the env")
            self.driver.find_element(By.TAG_NAME, 'canvas').click()
        logs = self.driver.get_log('browser')
        c = 0
        while not logs:
            # print(f"{c} logs not found sleeping...")
            sleep(0.001)
            logs = self.driver.get_log('browser')
            c += 1
            if c % 10 == 0:
                self.driver.find_element(By.TAG_NAME, 'canvas').click()
                logs = self.driver.get_log('browser')

        log = logs[-1]['message']
        # print(log)
        log = log.replace("http://localhost:4000/pakupaku/main.js 109:10 ", '')
        log = log.replace("http://localhost:4000/pakupaku/main.js 207:14 ", '')
        log = log.replace("http://localhost:4000/pakupaku/main.js 226:10 ", '')
        log = literal_eval(log)
        # log = literal_eval(logs[-1]['message'].replace("http://localhost:4000/pakupaku/main.js 109:10 ", ''))
        log = json.loads(log)
        # print(json.dumps(log, indent=4), len(log['dots']))
        return log



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_score = 1
        self.previous_dots = [{"x":11,"isPower":False}]*15
        self.reward_tracker = 0
        self.action_tracker = []
        self.score_tracker = [self.previous_score]
        self.previous_player_position = 40
        self.counter = 0
        self.enemy_close_counter = 0
        self.enemy_away_counter = 0
        self.power_up_enemy_close_counter = 0
        self.power_up_enemy_away_counter = 0
        self.scored_counter = 0
        self.dots_eaten = 0
        self.eaten_enemy = 0
        self.player_switching = 0
        self.did_not_thing = 0
        self.power_up_counter = 0
        self.switching_action = 0
        self.last_counter_when_dot_eaten = 0
        self.moving_towards_enemy = 0
        self.moving_against_enemy = 0
        self.enemy_close = 0

        self.log = self.get_browser_log(reset=True)

        self.obs = self.get_obs(self.log)
        return self.obs, {}

    def step(self, action):
        terminated = False
        done = False
        reward = 0
        desc = "|"

        action = action.item()
        
        log = self.log
        dots = log["dots"]
        score_diff = log['score'] - self.previous_score
        moving_towards_enemy = (
            log["player"]["vx"] > 0 and log["player"]["x"] < log["enemy"]["x"]
        ) or (log["player"]["vx"] < 0 and log["player"]["x"] > log["enemy"]["x"])
        pos_diff = abs(log['player']['x'] - log['enemy']['x'])


        if moving_towards_enemy:
            desc += "MovingTowardsEnemy|"
            self.moving_towards_enemy += 1

            if pos_diff < 20:
                desc += "EnemyClose|"
                self.enemy_close += 1
                if log['power_ticks'] > 0:
                    desc += "InPowerMode|"
                    reward += 1
                else:
                    desc += "InDangerMode|"
                    reward -= 1
            else:
                desc += "EnemyFar|"
                if log['power_ticks'] > 0:
                    desc += "InPowerMode|"
                    reward += 0.2
                else:
                    desc += "InDangerMode|"
                    reward -= 0.01
        else:
            desc += "MovingAgainstEnemy|"
            self.moving_against_enemy += 1
            if pos_diff < 20:
                desc += "EnemyClose|"
                self.enemy_close += 1
                if log['power_ticks'] > 0:
                    desc += "InPowerMode|"
                    reward -= 10
                else:
                    desc += "InDangerMode|"
                    reward += 0.1
            else:
                desc += "EnemyFar|"
                if log['power_ticks'] > 0:
                    desc += "InPowerMode|"
                    reward -= 1
                else:
                    desc += "InDangerMode|"
                    reward += 0.1


        if len(dots) != len(self.previous_dots):
            reward += 1
            self.dots_eaten += 1
            self.previous_dots = dots
            self.last_counter_when_dot_eaten = self.counter
            desc += "DotsEaten|"

        if score_diff > 1:
            reward += score_diff
            self.eaten_enemy += 1
            self.last_counter_when_dot_eaten = self.counter
            desc += "EnemyEaten|"


        if len(self.action_tracker) > 10 and self.action_tracker[-4:] == [self.switching_action] * 4:
            reward -= 1
            self.player_switching += 1
            desc += "KeepsSwitching|"
        
        if self.counter - self.last_counter_when_dot_eaten > 100:
            reward -= 1
            desc += "KeepsAvoidingEnemy"

        if action == self.switching_action:
            reward -= 0.85
            desc += "SwitchingNegativeReward"

        self.previous_score = log['score']
        self.reward_tracker += reward
        self.action_tracker.append(action)
        self.score_tracker.append(score_diff)
        info = {
            "reward": reward,
            "reward_tracker": self.reward_tracker,
            "score": log['score'],
            "game_ended": log['game_ended'],
            "player_position": log['player']['x'],
            "enemy_position": log['enemy']['x'],
            "counter": self.counter,
            "dots_eaten": self.dots_eaten,
            "eaten_enemy": self.eaten_enemy,
            "enemy_close_counter": self.enemy_close_counter,
            "enemy_away_counter": self.enemy_away_counter,
            "power_up_enemy_close_counter": self.power_up_enemy_close_counter,
            "power_up_enemy_away_counter": self.power_up_enemy_away_counter,
            "player_switching": self.player_switching,
            "did_not_thing": self.did_not_thing,
            "desc": desc,
            "moving_towards_enemy": moving_towards_enemy,
            "power_up_counter": self.power_up_counter,
        }
        # if log['power_ticks'] > 0:
        # current_datetime = str(int(datetime.utcnow().timestamp()*1000000))
        # filename = Path("imgs") / f"{current_datetime}-{desc}-{reward:.2f}.png"
        # Image.fromarray(self.obs).save(fp=filename, save_all=True, bitmap_format='png', )


        reward = max(min(reward, 1), -1)
        # print(log['game_ended'])

        if log['game_ended']:
            common_res = Counter(self.action_tracker).most_common(2)
            for c in common_res:
                if c[0] == self.switching_action:
                    info[f"Switching %"] = (c[1]/len(self.action_tracker))*100
                info[c[0]] = c[1]

            if log['score'] > 5:
                print(json.dumps(info, indent=4))
            terminated = True
            done = False
            return self.obs, reward, terminated, done, info

        self.counter += 1
        ActionChains(self.driver).send_keys(action).perform()
        self.log = self.get_browser_log()
        self.obs = self.get_obs(self.log)
        return self.obs, reward, terminated, done, info

    def close(self):
        self.driver.quit()


    def render(self):
        pass


def main():
    env = PakuPakuEnv()
    for _ in range(2):
        env.reset()
        # break
        for r in range(5000):
            # action = np.array([int(input("1 for same path, 0 to change direction>> "))])
            action = SystemRandom().choice(np.array([0, 1], dtype=np.int16))
            observation, reward, terminated, done, info = env.step(action)
            print(json.dumps(info, indent=4), done, terminated)
            if done or terminated:
                break
    sleep(10)
    env.close()


if __name__ == "__main__":
    from time import sleep
    from random import SystemRandom

    main()

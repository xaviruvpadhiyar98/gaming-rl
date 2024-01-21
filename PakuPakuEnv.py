import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread, Event
from uvicorn import Config, Server
import asyncio
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
from PIL import Image
from base64 import b64decode
from io import BytesIO
from collections import Counter
from time import sleep
import logging
from torch.utils.tensorboard import SummaryWriter
import json

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s %(threadName)-11s [%(filename)s:%(lineno)d %(funcName)5s()] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)



class ScreenshotRequest(BaseModel):
    screenshot: str
    score: int
    game_ended: bool
    player_position: float
    enemy_position: float
    power_ticks: float


class PakuPakuEnv(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, port=9000):
        super(PakuPakuEnv, self).__init__()
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=0, high=255, shape=(50, 100, 4), dtype=np.uint8
        )
        self.port = port

        # Selenium 
        options = ChromeOptions()
        options.binary_location = (
            Path.home() / "Softwares/thorium-browser_117.0.5938.157_amd64/thorium"
                ).as_posix()
        # options.add_argument("--auto-open-devtools-for-tabs")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        # options.add_argument("--headless")
        self.driver = Chrome(service=ChromeService(), options=options)


        # Fastapi
        self.app = FastAPI()
        self.origins = [
            "http://localhost:4000",
        ]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.post("/upload-screenshot")(self.get_data)

        # Uvicorn
        server_config = Config(
            self.app, host="127.0.0.1", port=self.port, lifespan="on", loop="asyncio", log_level="warning"
        )
        self.server = Server(server_config)


        # Threading
        self.request_event = Event()
        self.action_event = Event()
        self.thread = Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:
            sleep(0.1)
        self.request = None
        self.action = None
        # self.logger = SummaryWriter("tensorboard_log/ppo_1")



    def get_data(self, request: ScreenshotRequest) -> JSONResponse:
        # logger.info("New Incoming Request")
        self.request = request
        # logger.info(f"Got Request")
        self.request_event.set()
        # logger.info(f"{self.request_event.is_set()=}")
        # logger.info(f"{self.action_event.is_set()=}")
        self.action_event.wait(60)
        # logger.info(f"{self.action_event.is_set()=}")
        content = {"action": self.action}
        # logger.info(f"action is {content=}")
        
        self.request = None
        self.request_event.clear()
        # logger.info(f"{self.request_event.is_set()=}")
        self.action = None
        self.action_event.clear()
        # logger.info(f"{self.action_event.is_set()=}")
        return JSONResponse(content=content)

    def get_obs(self, record):
        b64_img = record.screenshot.replace("data:image/png;base64,", "")
        img = Image.open(BytesIO(b64decode(b64_img)))
        return np.asarray(img)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.driver.get("http://localhost:4000/?pakupaku")
        ActionChains(self.driver).send_keys(Keys.ENTER).perform()
        # logger.info(f"{self.request_event.is_set()=}")
        self.request_event.wait(60)
        request = self.request
        obs = self.get_obs(request)

        self.previous_score = request.score
        self.reward_tracker = 0
        self.action_tracker = ""
        self.score_tracker = [request.score]
        self.previous_player_position = request.player_position
        self.counter = 0
        self.enemy_close_counter = 0
        self.enemy_away_counter = 0
        self.power_up_enemy_close_counter = 0
        self.power_up_enemy_away_counter = 0
        self.scored_counter = 0
        self.dots_eaten = 0
        self.eaten_enemy = 0
        self.player_switching = 0


        return obs, {}
        

    def step(self, action):
        terminated = False
        done = False
        reward = 0
        desc = ""

        self.action = action.item()
        self.action_event.set()
        # logger.info(f"{self.action_event.is_set()=} {self.action=}")

        self.request = None
        self.request_event.clear()
        # logger.info(f"{self.request_event.is_set()=}")
        self.request_event.wait(60)
        request = self.request
        # logger.info(f"{self.request_event.is_set()=}")
        obs = self.get_obs(request)
        # logger.info((request.player_position, request.enemy_position, request.score, request.game_ended))



        score_diff = request.score - self.previous_score
        pos_diff = abs(request.player_position - request.enemy_position)


        if score_diff == 1:
            reward += 0.3
            self.dots_eaten += 1
            desc += "Dot Eaten. "
        
        if score_diff > 1:
            reward += 1
            self.eaten_enemy += 1
            desc += "Eaten Enemy. "
        
        if request.power_ticks > 0:
            if pos_diff < 8:
                reward += 1
                self.power_up_enemy_close_counter += 1
                desc += "PowerUp Enemy Close. "
            else:
                reward += 0.01
                self.power_up_enemy_away_counter += 1
                desc += "PowerUp Enemy Away. "
        else:
            if pos_diff < 13:
                reward -= 1
                self.enemy_close_counter += 1
                desc += "Enemy Close. "
            else:
                reward += 0.01
                self.enemy_away_counter += 1
                desc += "Enemy Away. "

        if len(self.action_tracker) > 10 and self.action_tracker[-5:] == '00000':
            reward -= 0.3
            self.player_switching += 1
            desc += "Keeps Switching. "


        # if pos_diff < 15:
        #     reward -= 1
        #     self.enemy_close_counter += 1
        #     desc += "Enemy Close. "

        # if pos_diff >= 15:
        #     reward += 0.05
        #     self.enemy_away_counter += 1
        #     desc += "Enemy Away. "






        # if len(self.score_tracker) > 200 and not request.game_ended:
        #     past_n_score = self.score_tracker[-100:]
        #     if max(past_n_score) == 0:
        #         reward -= 1
        #         desc += " Didn't earn any score in last 100 moves"

        # if len(self.action_tracker) > 10 and not request.game_ended:
        #     past_n_action = self.action_tracker[-10:]
        #     if max(past_n_action) == 0:
        #         reward -= 1
        #         desc += " keeps switching leading to same position"

        
        self.previous_score = request.score
        self.reward_tracker += reward
        self.action_tracker += str(action)
        self.score_tracker.append(score_diff)
        self.counter += 1
        info = {
            "reward": reward,
            "reward_tracker": self.reward_tracker,
            "score": request.score,
            "game_ended": request.game_ended,
            "player_position": request.player_position,
            "enemy_position": request.enemy_position,
            "counter": self.counter,
            "dots_eaten":self.dots_eaten,
            "eaten_enemy":self.eaten_enemy,
            "enemy_close_counter":self.enemy_close_counter,
            "enemy_away_counter":self.enemy_away_counter,
            "power_up_enemy_close_counter": self.power_up_enemy_close_counter,
            "power_up_enemy_away_counter": self.power_up_enemy_away_counter,
            "player_switching": self.player_switching,
            "desc": desc,
        }
        # print(info)
        if request.game_ended:
            reward -= 10
            common_res = Counter(self.action_tracker).most_common(2)
            for c in common_res:
                info[c[0]] = c[1]

            # self.logger.add_scalar("xgame/reward", reward)
            # self.logger.add_scalar("xgame/score", request.score)
            # self.logger.add_scalar("xgame/reward_tracker", self.reward_tracker)
            # self.logger.add_scalar("xgame/player_position", request.player_position)
            # self.logger.add_scalar("xgame/enemy_position", request.enemy_position)
            # self.logger.add_text("xgame/desc", desc)

            print(json.dumps(info, indent=4))
            terminated = True
            done = False

        reward = max(min(reward, 1), -1)

        
        return obs, reward, terminated, done, info


    def close(self):
        # self.logger.close()
        self.thread.join(timeout=1.0)
        self.driver.quit()
        self.server.should_exit = True
        while self.thread.is_alive():
            sleep(1)

    def render(self):
        pass

def main():
    env = PakuPakuEnv(port=8000)
    for _ in range(10):
        env.reset()
        for r in range(5000):
            action = SystemRandom().choice(np.array([1,], dtype=np.int16))
            observation, reward, terminated, done, info = env.step(action)
            if done or terminated:
                break
    
    sleep(10)
    env.close()

if __name__ == "__main__":
    from time import sleep
    from random import SystemRandom
    main()



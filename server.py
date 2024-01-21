from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from time import sleep


app = FastAPI()

ENV_RECORD = Path("env_record")
ENV_RECORD.mkdir(parents=True, exist_ok=True)


origins = [
    "http://localhost:4000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenshotRequest(BaseModel):
    screenshot: str
    score: int
    gameEnded: bool
    playerPosition: float
    enemyPosition: float


@app.post("/upload-screenshot")
def upload_file(request: ScreenshotRequest):
    (ENV_RECORD / "record1").write_text((request.model_dump_json()))
    action_file = ENV_RECORD / "action1"
    while not action_file.exists():
        sleep(0.01)
    action = action_file.read_text().strip()
    if action == "":
        return {"action": 1}
    return {"action": int(action)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="localhost", port=8000, reload=False)

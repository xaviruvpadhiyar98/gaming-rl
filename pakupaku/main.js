title = "PAKU PAKU";

description = `
[Tap] Turn
`;

characters = [
  `
    llll
   lll
  lll
  lll
   lll
    llll
  `,
  `
    lll
   lllll
  lll
  lll
   lllll
    lll
  `,
  `
    ll
   llll
  llllll
  llllll
   llll
    ll
  `,
  `
    lll
   l l l
   llll
   llll
  llll
  l l l
  `,
  `
    lll
   l l l
   llll
   llll
   llll
   l l
  `,
  `
  ll
  ll
  `,
  `
   ll
  llll
  llll
   ll
  `,
  `
    l l
  
  
  
  `,
];

options = {
  theme: "dark",
  viewSize: { x: 100, y: 50 },
  isPlayingBgm: false,
  isReplayEnabled: false,
  seed: 20,
};

/** @type {{x: number, vx: number}} */
let player;
/** @type {{x: number, eyeVx: number}} */
let enemy;
/** @type {{x: number, isPower: boolean}[]} */
let dots;
let powerTicks;
let animTicks;
let multiplier;
let isGamePaused = false;
let gameEnded = false;




function send_screenshot() {
  isGamePaused = true; // Pause the game when the request is made
  const screenshotData = document.getElementsByTagName('canvas')[0].toDataURL('image/png');
  const jsonString = JSON.stringify({
    screenshot: screenshotData,
    score: score,
    gameEnded: gameEnded
  })

  fetch('http://localhost:8000/upload-screenshot', {
    method: 'POST',
    body: JSON.stringify({
      screenshot: screenshotData,
      score: score,
      gameEnded: gameEnded,
      playerPosition: player.x,
      enemyPosition: enemy.x,
    }),
    headers: {
      'Content-Type': 'application/json',
    },
  })
  .then(response => response.json())
  .then(data => {
    // Handle response here
    // console.log(data);
    if (data.action == 0) {
      player.vx *= -1;
    }
    isGamePaused = false; // Resume the game after handling the response
  })
  .catch(error => {
    // console.error('Error:', error);
    isGamePaused = false; // Resume the game even if there's an error
  });
}


function update() {
  if (!ticks) {
    // console.log("starting here")
    player = { x: 40, vx: 1 };
    enemy = { x: 100, eyeVx: 0 };
    multiplier = 0;
    addDots();
    powerTicks = animTicks = 0;
  }
  if (isGamePaused) {
    return; // Exit the function early if the game is paused
  }
  animTicks += difficulty;
  color("black");
  text(`x${multiplier}`, 3, 9);
  player.x += player.vx * 0.5 * difficulty;
  if (player.x < -3) {
    player.x = 103;
  } else if (player.x > 103) {
    player.x = -3;
  }
  color("blue");
  rect(0, 23, 100, 1);
  rect(0, 25, 100, 1);
  rect(0, 34, 100, 1);
  rect(0, 36, 100, 1);
  color("green");
  const ai = floor(animTicks / 7) % 4;
  char(addWithCharCode("a", ai === 3 ? 1 : ai), player.x, 30, {
    // @ts-ignore
    mirror: { x: player.vx },
  });
  remove(dots, (d) => {
    color(
      d.isPower && floor(animTicks / 7) % 2 === 0 ? "transparent" : "yellow"
    );
    const c = char(d.isPower ? "g" : "f", d.x, 30).isColliding.char;
    if (c.a || c.b || c.c) {
      if (d.isPower) {
        // play("jump");
        if (enemy.eyeVx === 0) {
          powerTicks = 120;
        }
      } else {
        // play("hit");
      }
      addScore(multiplier);
      return true;
    }
  });
  const evx =
    enemy.eyeVx !== 0
      ? enemy.eyeVx
      : (player.x > enemy.x ? 1 : -1) * (powerTicks > 0 ? -1 : 1);
  enemy.x = clamp(
    enemy.x +
      evx *
        (powerTicks > 0 ? 0.25 : enemy.eyeVx !== 0 ? 0.75 : 0.55) *
        difficulty,
    0,
    100
  );
  if ((enemy.eyeVx < 0 && enemy.x < 1) || (enemy.eyeVx > 0 && enemy.x > 99)) {
    enemy.eyeVx = 0;
  }
  color(
    powerTicks > 0
      ? powerTicks < 30 && powerTicks % 10 < 5
        ? "black"
        : "blue"
      : enemy.eyeVx !== 0
      ? "black"
      : "red"
  );
  const c = char(
    enemy.eyeVx !== 0 ? "h" : addWithCharCode("d", floor(animTicks / 7) % 2),
    enemy.x,
    30,
    {
      // @ts-ignore
      mirror: { x: evx },
    }
  ).isColliding.char;
  if (enemy.eyeVx === 0 && (c.a || c.b || c.c)) {
    if (powerTicks > 0) {
      // play("powerUp");
      addScore(10 * multiplier, enemy.x, 30);
      enemy.eyeVx = player.x > 50 ? -1 : 1;
      powerTicks = 0;
      multiplier++;
    } else {
      // play("explosion");
      end();
      gameEnded = true;
    }
  }
  powerTicks -= difficulty;
  if (dots.length === 0) {
    // play("coin");
    addDots();
  }

  send_screenshot();
}



function addDots() {
  let pi = player.x > 50 ? rndi(1, 6) : rndi(10, 15);
  dots = times(16, (i) => ({ x: i * 6 + 5, isPower: i === pi }));
  multiplier++;
}

# RL for Autonomous Robot Navigation
### Reinforcement Learning for Reduced Continuous Control in Assistive Navigation

**Stephanie Pavon — NYU Tandon School of Engineering**

---

## Motivation

In assistive robotics, users with limited dexterity often struggle to provide continuous, precise control inputs. This project explores whether reinforcement learning can eliminate that burden. The user specifies a destination, and the robot autonomously handles obstacle avoidance, path adjustment, and low-level motion.

Using a TurtleBot3 simulated in Gazebo and controlled through ROS2, a PPO policy was trained to navigate to random goals while avoiding randomly placed obstacles. No hand-coded rules, no pre-programmed paths.

---

## Results

| Metric | Value |
|---|---|
| Total training steps | 563,200 |
| Training time (CPU only) | 5.07 days (~122 hours) |
| Final smoothed reward | 113.3 |
| Evaluation success rate | 55% (11/20 goals reached) |
| Best demo run | 5/5 goals, avg reward 201.77, avg 89.6 steps |

---

## System Architecture

```
Gazebo Simulation
      ↓
ROS2 Topics  (/scan · /odom · /cmd_vel)
      ↓
RL Environment  (turtlebot_gym_env.py — Gym-style wrapper)
      ↓
PPO Algorithm  (Stable-Baselines3 · MlpPolicy · CPU)
```

The RL environment acts as a bridge between ROS2's asynchronous event-based topics and the synchronous `reset()` / `step()` interface that Stable-Baselines3 expects.

---

## MDP Formulation

### State Space (10 values)
| Index | Value |
|---|---|
| 0 | Normalized distance to goal (0–1, max arena diagonal = 6m) |
| 1 | Relative angle to goal (radians, wrapped to −π to π) |
| 2–9 | Minimum LiDAR distance in 8 sectors: Front (0°), Front-Left (45°), Left (90°), Back-Left (135°), Back (180°), Back-Right (225°), Right (270°), Front-Right (315°) |

### Action Space (4 discrete actions)
| Action | Command |
|---|---|
| 0 — Forward | linear.x = 0.3 m/s |
| 1 — Turn Left | angular.z = +0.5 rad/s |
| 2 — Turn Right | angular.z = −0.5 rad/s |
| 3 — Stop | linear.x = 0, angular.z = 0 |

### Reward Function
| Signal | Value |
|---|---|
| Goal reached (within 20 cm) | +200 |
| Collision (within 18 cm) | −100 |
| Progress toward goal | +15 × (prev_distance − curr_distance) |
| Time penalty (every step) | −0.1 |
| Extra penalty for stop action | −0.1 |
| Heading bonus (forward action only) | up to +0.2 × (1 − \|angle\| / π) |

> **Note on heading bonus:** Originally applied to all actions, which caused reward hacking, the robot learned to spin in place, collecting small bonuses instead of navigating. Restricting it to forward motion only (action 0) fixed this.

### Termination Conditions
- Collision detected (any of 5 forward-facing LiDAR sectors < 18 cm)
- Goal reached (distance < 20 cm)
- Max steps reached (750 steps — truncation)

---

## Training Journey

### Part 1 — Fixed goal, fixed obstacles
- 5-state observation, 3 LiDAR sectors, 3 collision directions, 500 step max
- Robot navigates to one fixed goal position
- **Converged at +52** — reward hacking (spinning for heading bonus) and diagonal blind spots limited learning

### Part 2 Step 1 — 8 random safe goals + 4 improvements
- Expanded observation: 5 → 10 states
- LiDAR sectors: 3 → 8 directions
- Normalized distance (raw meters → 0–1 scale)
- Collision detection: 3 → 5 directions (added front-left, front-right)
- Max steps: 500 → 750
- Fixed reward hacking by restricting heading bonus to forward action only
- **Peaked at 110+, averaged 90+**

### Part 2 Step 2 — Truly random goals + curriculum learning
- Goals randomly placed anywhere in arena each episode
- Curriculum: `difficulty = min(episode / 2000, 1.0)` where the arena bounds expand gradually, so early episodes start with nearby goals
- 3-check goal placement: spawn distance · cylinder distance · arena boundary
- **Discovered boundary bug:** difficulty scaling used `max()` instead of `min()`, pushing goals outside arena walls at high episode counts — reward dropped from 130 → 50 by the end of training
- **Peaked at +127, evaluation: 6/10 goals, avg reward 98.18**

### Part 2 Step 3 — Random obstacles + boundary bug fix
- Fixed boundary bug (changed `max()` to `min()` for arena bound caps)
- 9 cylinders repositioned every episode via Gazebo's `/set_entity_state` service
- Obstacles placed before goal each reset, the goal picker avoids new cylinder positions
- Training continued from Step 2 checkpoint (307k steps) — no reset
- At 307k: random obstacles introduced → reward dipped to ~50, then oscillated and recovered
- **Final smoothed reward: 113 · Evaluation (20 episodes): 11/20 goals (55%)**

---

## Curriculum Learning

```python
difficulty = min(self.episode_count / 2000.0, 1.0)

x_min = -1.5
x_max = min(-1.5 + difficulty * 3.0, 1.5)   # hard cap at arena wall
y_min = max(-difficulty * 1.9, -1.9)          # hard cap at arena wall
y_max = min(difficulty * 1.9, 1.9)            # hard cap at arena wall

x_max = max(x_max, 0.0)   # always some reachable area early on
y_max = max(y_max, 0.5)
```

Episode count persists across training sessions via `episode_count.txt`.

---

## Episode Reset Flow

Each episode reset follows this sequence:

1. Stop the robot
2. Call `/reset_simulation` — robot teleports to spawn, cylinders snap to SDF defaults
3. Wait for Gazebo to settle (0.5s)
4. **Teleport all 9 cylinders to new random positions** via `set_entity_state`
5. Pick a new random goal (avoiding the spawn zone and cylinder positions)
6. Wait for a fresh LiDAR scan
7. Return first observation

---

## Project Structure

```
rl-obstacle-navigation/
├── turtlebot_gym_env.py    # RL environment (Gym-style ROS2 wrapper)
├── train_ppo.py            # PPO training script (auto-resumes from checkpoint)
├── evaluate.py             # Evaluation script (pre-selected demo goals)
├── episode_count.txt       # Persists episode count across training sessions
└── ppo_checkpoints/        # Saved model checkpoints every 10k steps
```

---

## How to Run

### 1. Launch Gazebo simulation
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### 2. Activate Python environment
```bash
source ~/sb3_env/bin/activate
```

### 3. Train
```bash
python3 train_ppo.py
```
Trains for 50,000 steps per session. Auto-resumes from `ppo_turtlebot.zip` if it exists. Saves a checkpoint every 10k steps to `ppo_checkpoints/`. Run again to add another 50k steps.

### 4. Evaluate
```bash
python3 evaluate.py
```
Loads the trained model and runs 5 pre-selected episodes to demonstrate goal reached, max steps, and collision outcomes. Obstacles remain random in each episode.

### 5. Monitor training (TensorBoard)
```bash
tensorboard --logdir ./ppo_logs/
```

---

## Limitations

- **Coarse LiDAR (8 sectors):** Each sector reports only the minimum distance; the robot cannot detect narrow openings between cylinders and gets stuck in dense clusters
- **Memoryless policy (MLP):** Sees only the current timestep, no memory of past moves, so it can repeat the same failed direction

## Future Work

- **Full 360° raw LiDAR scan as input** — feed all 360 beams directly to detect gap shapes and opening widths that sector minimums compress away
- **LSTM recurrent policy** — hidden state across timesteps so the robot can recognize it is stuck and try a new direction
- **Assistive interface** — simple user-triggered input (mouth control, gesture, switch) to initiate navigation while the robot handles all low-level motion autonomously

---

## Technologies

- ROS2 (Humble)
- Gazebo
- Python 3
- Stable-Baselines3 (PPO)
- Gymnasium
- NumPy
- TensorBoard

---

## Repository

- `main` branch — final trained model and evaluation
- `part2` branch — Part 2 Step 3 development

[github.com/spavonnn/rl-obstacle-navigation](https://github.com/spavonnn/rl-obstacle-navigation)

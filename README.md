# Evaluating Reinforcement Learning for Reduced Continuous Control in Assistive Navigation 

## Overview
This project explores the use of reinforcement learning (RL) to simplify continuous control in assistive robotic navigation tasks.
Using a TurtleBot simulated in Gazebo and controlled through ROS2 communication interfaces, the goal is to learn navigation behaviors that require minimal continuous control effort while still achieving safe and effective obstacle avoidance.

This project focuses on designing a structured RL environment that integrates robot perception, control, and reward-driven learning to evaluate how RL can improve assistive navigation scenarios.
This setup enables the evaluation of reinforcement learning methods for reducing control complexity in assistive navigation tasks.

---

## Motivation 
In assistive robotics, users may have a limited ability to provide continuous, precise control inputs. This project investigates whether reinforcement learning can reduce the need for continuous control by enabling the robot to autonomously handle navigation decisions such as obstacle avoidance and goal-directed movement.

---

## System Architecture 

This system integrates simulation, robotics middleware, and reinforcement learning components:

Gazebo (simulation) --> ROS2 Topics (communication) --> RL Environment (Gym-Style) --> RL Algorithm - Policy Learning (PPO)

- **Gazebo**: Simulates the robot and environment with obstacles
- **ROS2**: Handles communication through topics such as '/scan', '/odom', and '/cmd_vel'
- **RL Environment**: Wraps sensor data and control into a Gym-style interface
- **Policy Learning (future work)**: PPO will be used to learn navigation strategies

The RL environment acts as a bridge between ROS2 and the learning algorithm by converting sensor data into observations and control commands into executable robot actions. 

--- 

## Features

- ROS2-based integration in TurtleBot simulation
- Custom Gym-style reinforcement learning environment
- LiDAR-based perception for obstacle avoidance
- Goal-oriented navigation behavior
- ROS2-to-RL interface for real-time sensor and control integration. A modular pipeline connecting ROS2 and RL.
- Test script for validating environment behavior

---

# Observation Space

The agent observes a state representation:

- Distance to goal
- Relative angle to goal
- LiDAR readings:
    - Front
    - Left
    - Right
  
Example:

```python
obs = [distance_to_goal, relative_angle, front, left, right]

```

## Action Space

The agent outputs continuous control commands:

- Linear Velocity
- Angular Velocity

These actions are applied to the robot through ROS2 velocity commands.

## Reward Design
The reward function encourages safe and efficient navigation:

- Positive reward for moving toward the goal
- Penalty for proximity to obstacles
- Penalty for collisions
- Reward for reaching the goal

This design implements smooth motion, goal-directed behavior, and safe obstacle avoidance.

## Project Structure
```
rl-obstacle-navigation/
├── turtlebot_rl_env/
│   ├── __init__.py
│   └── turtlebot_gym_env.py    # Main RL environment
│
├── scripts/
│   └── test_env.py             # Test script for environment
│
├── resource/
├── test/
├── package.xml
├── setup.py
├── setup.cfg
└── .gitignore

```
---

## How To Run
## 1. Build the ROS2 workspace

```bash
cd ~/rl_ws
colcon build
source install/setup.bash
```
---

## 2. Launch the simulation
Start your TurtleBot Gazebo simulation environment.

---
## 3. Run the environment test

```bash
cd ~/rl_ws/src/turtlebot_rl_env
python3 scripts/test_env.py
```
This will: 
- Initialize the environment
- Execute actions
- Print observations, rewards, and termination conditions

---

## Current Status

- RL environment implemented and functional
- ROS2-RL interface completed
- Observation and action spaces defined
- Reward function designed and tested
- Environment validated using a test script

---

## Future Work

- Integrate PPO using Stable-Baselines3
- Train and evaluate navigation policy
- Evaluate performance under different obstacle configurations
- Incorporate additional sensors (e.g., camera) for improved perception
- Explore assistive navigation using a simple user-triggered input, with potential extension to accessible interfaces such as mouth-controlled or gesture-based signals

---
## Technologies Used
- ROS2
- Gazebo
- Python
- NumPy
- Gymnasium concept
- SB3

---
## Author

Stephanie Pavon 

NYU Tandon School of Engineering 

---

## Notes

This project is part of a graduate-level exploration into reinforcement learning for assistive robotics, intending to reduce continuous control demands during navigation tasks. In particular, it is motivated by improving accessibility for individuals with limited dexterity, where a simple user-triggered input can initiate navigation while the robot autonomously handles low-level motion, obstacle avoidance, and path adjustment.

---

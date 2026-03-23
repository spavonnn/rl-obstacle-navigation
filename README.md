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

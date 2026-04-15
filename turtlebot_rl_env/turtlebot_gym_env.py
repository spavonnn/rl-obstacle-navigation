# imports
import gymnasium as gym
import numpy as np
import rclpy
import time
import random

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from tf_transformations import euler_from_quaternion


class TurtlebotGymEnv(gym.Env):

	def __init__(self):
		super().__init__()

		# initializing ROS only once
		if not rclpy.ok():
			rclpy.init()

		# creating ROS node
		self.node = rclpy.create_node("turtlebot_gym_env")

		# publisher for robot velocity commands
		self.cmd_vel_pub = self.node.create_publisher(
			Twist,
			"/cmd_vel",
			10
		)

		# subscriber for odometry
		self.odom_sub = self.node.create_subscription(
			Odometry,
			"/odom",
			self.odom_callback,
			10
		)

		# subscriber for lidar scan
		self.scan_sub = self.node.create_subscription(
			LaserScan,
			"/scan",
			self.scan_callback,
			10
		)

		# /reset_simulation service client
		self.reset_sim_client = self.node.create_client(
			Empty,
			"/reset_simulation"
		)

		# robot pose values (in odometry frame)
		self.robot_x   = 0.0
		self.robot_y   = 0.0
		self.robot_yaw = 0.0

		# -------------------------------------------------------
		# PART 2 - RANDOMIZED GOAL POSITIONS FROM SET LIST THEN LATER ON TRUE RANDOM GOALS WITH OBSTACLE CHECKING + RANDOM OBSTACLES
		# -------------------------------------------------------
		# Instead of a fixed goal, a new random goal is picked
		# at the start of every episode from a list of safe
		# positions that are verified to be clear of obstacles
		# and walls.
		#
		# COORDINATE SYSTEM:
		# All positions below are in odom frame.
		# odom = gazebo_world - spawn_offset
		# spawn_offset = (-2.0, -0.5)
		#
		# ARENA BOUNDARIES (odom frame, with 0.3m wall buffer):
		# x range: 0.5 to 3.5
		# y range: -1.4 to 2.4
		#
		# CYLINDER POSITIONS (odom frame, radius = 0.15m):
		# (0.9, -0.6), (0.9, 0.5), (0.9, 1.6)
		# (2.0, -0.6), (2.0, 0.5), (2.0, 1.6)
		# (3.1, -0.6), (3.1, 0.5), (3.1, 1.6)
		#
		# Each safe goal position below was chosen to be:
		# - inside the arena boundaries
		# - at least 0.5m away from every cylinder
		# - at least 0.5m away from the spawn position (0,0)
		# -------------------------------------------------------

		# list of safe goal positions in odom frame
		self.safe_goal_positions = [
			(1.5,  0.5),   # open gap between cylinder rows (part 1 goal)
			(1.5, -1.0),   # open area bottom right of arena
			(1.5,  2.0),   # open area top right of arena
			(0.5,  0.0),   # close open area near spawn
			(0.5,  1.5),   # open area top left of arena
			(0.5, -1.0),   # open area bottom left of arena
			(2.5, -1.0),   # open area bottom middle of arena
			(2.5,  2.0),   # open area top middle of arena
		]

		# current goal — gets set randomly in reset()
		self.goal_x = self.safe_goal_positions[0][0]
		self.goal_y = self.safe_goal_positions[0][1]

		# lidar storage
		self.lidar_ranges = None
		self.new_scan     = False

		# previous distance for progress reward
		self.prev_distance_to_goal = None

		# step counter for max episode length truncation
		# if robot hasn't collided or reached the goal after this many steps,
		# end the episode to prevent wandering forever
		self.current_step = 0
		self.max_steps    = 500

		# action space:
		# 0 = forward
		# 1 = turn left
		# 2 = turn right
		# 3 = stop
		self.action_space = gym.spaces.Discrete(4)

		# observation space: 5 states
		# [distance_to_goal, relative_angle, front, left, right]
		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(5,),
			dtype=np.float32
		)

	def _pick_random_goal(self):
		"""
		Picks a random goal position from the safe_goal_positions list.
		Makes sure the new goal is not the same as the current goal
		so the robot always gets a different target each episode.
		"""
		# get all positions except the current one
		other_positions = [
			pos for pos in self.safe_goal_positions
			if pos != (self.goal_x, self.goal_y)
		]

		# pick a random one
		new_goal = random.choice(other_positions)
		self.goal_x = new_goal[0]
		self.goal_y = new_goal[1]

		self.node.get_logger().info(
			f"New goal set to odom ({self.goal_x}, {self.goal_y})"
		)

	def reset(self, seed=None, options=None):
		# standard gymnasium reset
		super().reset(seed=seed)

		# step 1 - stop the robot
		cmd = Twist()
		cmd.linear.x  = 0.0
		cmd.angular.z = 0.0
		self.cmd_vel_pub.publish(cmd)

		# step 2 - call /reset_simulation to teleport robot back to spawn
		self.reset_sim_client.wait_for_service(timeout_sec=5.0)
		future = self.reset_sim_client.call_async(Empty.Request())
		while not future.done():
			rclpy.spin_once(self.node, timeout_sec=0.05)

		# step 3 - wait for Gazebo to finish the reset
		time.sleep(0.5)

		# step 4 - stop again after reset
		self.cmd_vel_pub.publish(cmd)

		# step 5 - pick a new random goal for this episode
		self._pick_random_goal()

		# step 6 - wait for a fresh LiDAR scan from the new spawn position
		self.new_scan = False
		while not self.new_scan:
			rclpy.spin_once(self.node, timeout_sec=0.01)

		# step 7 - get first observation and return it
		observation = self._get_obs()
		self.prev_distance_to_goal = observation[0]

		# reset step counter
		self.current_step = 0

		info = {}
		return observation, info

	def step(self, action):
		# create velocity command
		cmd = Twist()

		# action 0 = forward
		if action == 0:
			cmd.linear.x  = 0.3
			cmd.angular.z = 0.0

		# action 1 = turn left
		elif action == 1:
			cmd.linear.x  = 0.0
			cmd.angular.z = 0.5

		# action 2 = turn right
		elif action == 2:
			cmd.linear.x  = 0.0
			cmd.angular.z = -0.5

		# action 3 = stop
		elif action == 3:
			cmd.linear.x  = 0.0
			cmd.angular.z = 0.0

		# publish action
		self.new_scan = False
		self.cmd_vel_pub.publish(cmd)

		# wait for new sensor update
		while not self.new_scan:
			rclpy.spin_once(self.node, timeout_sec=0.01)

		# get new observation
		observation = self._get_obs()

		# unpacking observation values
		distance_to_goal, relative_angle, front, left, right = observation

		# thresholds
		collision_distance = 0.12  # 12cm — matches robot body size
		goal_distance      = 0.20

		# collision check
		collision = bool(
			front < collision_distance or
			left  < collision_distance or
			right < collision_distance
		)

		# check goal reached
		goal_reached = bool(distance_to_goal < goal_distance)

		# episode ends if collision or goal is reached
		terminated = bool(collision or goal_reached)

		# increment step counter
		self.current_step += 1

		# truncate episode if max steps reached
		truncated = bool(self.current_step >= self.max_steps)

		# log how the episode ended for debugging
		if terminated or truncated:
			if collision:
				self.node.get_logger().info("Episode ended due to COLLISION.")
			elif goal_reached:
				self.node.get_logger().info("Episode ended with GOAL REACHED!")
			else:
				self.node.get_logger().info("Episode ended due to MAX STEPS reached.")

		# computing reward
		reward = self._compute_reward(
			observation,
			collision,
			goal_reached,
			action
		)

		# updating previous distance after reward is calculated
		self.prev_distance_to_goal = distance_to_goal

		# extra debug info
		info = {
			"collision":    collision,
			"goal_reached": goal_reached,
			"step":         self.current_step,
			"goal":         (self.goal_x, self.goal_y)
		}

		return observation, reward, terminated, truncated, info

	def _compute_reward(self, observation, collision, goal_reached, action):
		# unpack observation
		distance_to_goal, relative_angle, front, left, right = observation

		reward = 0.0

		# big negative reward for collision
		if collision:
			reward -= 100.0

		# big positive reward for reaching goal
		if goal_reached:
			reward += 100.0

		# reward progress toward goal
		if self.prev_distance_to_goal is not None:
			progress = self.prev_distance_to_goal - distance_to_goal
			reward += 8.0 * progress

		# small time penalty so robot doesn't idle
		reward -= 0.1

		# small penalty for doing nothing
		if action == 3:
			reward -= 0.1

		# penalty if obstacle is too close in front
		if front < 0.20:
			reward -= 1.0

		return reward

	def odom_callback(self, msg):
		self.robot_x = msg.pose.pose.position.x
		self.robot_y = msg.pose.pose.position.y

		q    = msg.pose.pose.orientation
		quat = [q.x, q.y, q.z, q.w]

		roll, pitch, yaw = euler_from_quaternion(quat)
		self.robot_yaw = yaw

	def scan_callback(self, msg):
		self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)
		self.new_scan     = True

	def _get_obs(self):
		# distance from robot to goal
		dx = self.goal_x - self.robot_x
		dy = self.goal_y - self.robot_y
		distance_to_goal = np.sqrt(dx**2 + dy**2)

		# angle to goal wrapped to [-pi, pi]
		goal_angle     = np.arctan2(dy, dx)
		relative_angle = goal_angle - self.robot_yaw
		relative_angle = np.arctan2(
			np.sin(relative_angle),
			np.cos(relative_angle)
		)

		# lidar sector values
		if self.lidar_ranges is None:
			front = 10.0
			left  = 10.0
			right = 10.0
		else:
			ranges = np.nan_to_num(
				self.lidar_ranges,
				nan=10.0,
				posinf=10.0,
				neginf=10.0
			)

			n = len(ranges)
			k = 10  # sector half-width in beams

			front_sector = np.concatenate((ranges[:k], ranges[-k:]))
			left_center  = n // 4
			right_center = 3 * n // 4
			left_sector  = ranges[left_center  - k : left_center  + k]
			right_sector = ranges[right_center - k : right_center + k]

			front = float(np.nanmin(front_sector))
			left  = float(np.nanmin(left_sector))
			right = float(np.nanmin(right_sector))

		return np.array([
			distance_to_goal,
			relative_angle,
			front,
			left,
			right,
		], dtype=np.float32)

	def close(self):
		cmd = Twist()
		cmd.linear.x  = 0.0
		cmd.angular.z = 0.0
		self.cmd_vel_pub.publish(cmd)
		self.node.destroy_node()
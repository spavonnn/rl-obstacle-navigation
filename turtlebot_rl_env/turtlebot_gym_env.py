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
		self.max_steps    = 750

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
			shape=(10,),
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
		distance_to_goal, relative_angle, front, front_left, left, back_left, back, back_right, right, front_right = observation

		# thresholds
		collision_distance = 0.18  # 18cm — matches robot body size
		goal_distance      = 0.20

		# collision check
		collision = bool(
			front < collision_distance or
			front_left < collision_distance or
			front_right < collision_distance or
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
		distance_to_goal, relative_angle, front, front_left, left, back_left, back, back_right, right, front_right = observation

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
		
		# small reward for facing toward the goal
		heading_bonus = 0.3 * (1.0 - abs(relative_angle) / np.pi)
		reward += heading_bonus
		

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

		# normalize distance to 0-1 range using max arena diagonal
		max_distance = 6.0 # approximate max distance across the arena
		distance_normalized = float(np.clip(distance_to_goal / max_distance, 0.0, 1.0))

		# angle to goal wrapped to [-pi, pi]
		goal_angle     = np.arctan2(dy, dx)
		relative_angle = goal_angle - self.robot_yaw
		relative_angle = float(np.arctan2(
			np.sin(relative_angle),
			np.cos(relative_angle)
		))

		# lidar sector values
		# expanded from 3 sectors to 8 sectors for better obstacle awareness
		# this will help the robot navigate around obstacles more effectively

		if self.lidar_ranges is None:
			front = 10.0
			front_left = 10.0
			left  = 10.0
			back_left = 10.0
			back = 10.0
			back_right = 10.0
			right = 10.0
			front_right = 10.0

		else:
			ranges = np.nan_to_num(
				self.lidar_ranges,
				nan=10.0,
				posinf=10.0,
				neginf=10.0
			)

			n = len(ranges) # 360 beams
			k = 10  # sector half-width in beams

			# 8 sectors evenly spaced around 360 degrees
			# each sector center is 45 degrees apart
			# sector centers at: 0 (front), 45 (front-left), 90 (left), 135 (back-left), 180 (back), 225 (back-right), 270 (right), 315 (front-right)

			
			front_center = 0                    # 0 degrees
			front_left_center  = n // 8			# 45 degrees
			left_center       = n // 4			# 90 degrees
			back_left_center  = 3 * n // 8		# 135 degrees
			back_center       = n // 2			# 180 degrees
			back_right_center = 5 * n // 8		# 225 degrees
			right_center      = 3 * n // 4		# 270 degrees
			front_right_center = 7 * n // 8		# 315 degrees


			# front sector wraps around array start/end, so we concatenate the end and start of the ranges array
			front_sector = np.concatenate((ranges[:k], ranges[-k:]))

			# all other sectors
			front_left_sector  = ranges[front_left_center - k : front_left_center + k]
			left_sector       = ranges[left_center - k : left_center + k]
			back_left_sector  = ranges[back_left_center - k : back_left_center + k]
			back_sector       = ranges[back_center - k : back_center + k]
			back_right_sector = ranges[back_right_center - k : back_right_center + k]
			right_sector      = ranges[right_center - k : right_center + k]
			front_right_sector = ranges[front_right_center - k : front_right_center + k]

			# take minimum distance in each sector as the sector value
			front = float(np.nanmin(front_sector))
			front_left = float(np.nanmin(front_left_sector))
			left = float(np.nanmin(left_sector))
			back_left = float(np.nanmin(back_left_sector))
			back = float(np.nanmin(back_sector))
			back_right = float(np.nanmin(back_right_sector))
			right = float(np.nanmin(right_sector))
			front_right = float(np.nanmin(front_right_sector))

		return np.array([
			distance_normalized,     # normalized 0-1
			relative_angle,			 # angle to goal in radians, wrapped to [-pi, pi]
			front,			 		 # 0 degrees
			front_left,			     # 45 degrees
			left,				     # 90 degrees
			back_left,			     # 135 degrees
			back,				     # 180 degrees
			back_right,		     	 # 225 degrees
			right,				     # 270 degrees
			front_right			     # 315 degrees
		], dtype=np.float32)

	def close(self):
		cmd = Twist()
		cmd.linear.x  = 0.0
		cmd.angular.z = 0.0
		self.cmd_vel_pub.publish(cmd)
		self.node.destroy_node()
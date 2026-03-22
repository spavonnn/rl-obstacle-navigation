#imports
import gymnasium as gym
import numpy as np
import rclpy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
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

		# robot pose values
		self.robot_x = 0.0
		self.robot_y = 0.0
		self.robot_yaw = 0.0

		# goal position
		self.goal_x = 2.0
		self.goal_y = 0.0

		# lidar storage
		self.lidar_ranges = None
		self.new_scan = False

		# previous distance for progress reward
		self.prev_distance_to_goal = None


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

	def reset(self, seed=None, options=None):
		# reset simulation - standard gymnasium reset
		super().reset(seed=seed)

		# stop robot at the start of episode
		cmd = Twist()
		cmd.linear.x = 0.0
		cmd.angular.z = 0.0
		self.cmd_vel_pub.publish(cmd)

		# wait until at least one new scan arrives
		self.new_scan = False
		while not self.new_scan:
			rclpy.spin_once(self.node, timeout_sec=0.01)

		# get first real observation
		observation = self._get_obs()

		# saving distance for reward progress calculation
		self.prev_distance_to_goal = observation[0]

		info = { }
		return observation, info

	def step(self, action):
		# create velocity command
		cmd = Twist()

		# choosing robot actions:

		# action 0 = forward
		if action == 0:
			cmd.linear.x = 0.2
			cmd.angular.z = 0.0

		# action 1 = turn left
		elif action == 1:
			cmd.linear.x = 0.0
			cmd.angular.z = 0.5

		# action 2 = turn right
		elif action == 2:
			cmd.linear.x = 0.0
			cmd.angular.z = -0.5

		# action 3 = stop
		elif action == 3:
			cmd.linear.x = 0.0
			cmd.angular.z = 0.0


		# publish action
		self.new_scan = False
		self.cmd_vel_pub.publish(cmd)

		# wait for new sensor update
		while not self.new_scan:
			rclpy.spin_once(self.node, timeout_sec = 0.01) # process ROS messages until a new LiDAR scan callback updates self.new_scan

		# get new observation
		observation = self._get_obs()

		# unpacking observation values
		distance_to_goal, relative_angle, front, left, right = observation

		# thresholds
		collision_distance = 0.12
		goal_distance = 0.20

		# collision check
		collision = (
			front < collision_distance or
			left < collision_distance or
			right < collision_distance
		)

		# check goal reached
		goal_reached = distance_to_goal < goal_distance

		# episode ends if collision or goal is reached
		terminated = collision or goal_reached

		# no time-limit truncation yet
		truncated = False

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
			"collision": collision,
			"goal_reached": goal_reached
		}

		return observation, reward, terminated, truncated, info

	def _compute_reward(self, observation, collision, goal_reached, action):
		# collision penalty
		# progress toward goal
		# time penalty
		# goal reach and safe stopping bonus

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
			reward += 10.0 * progress

		# small penalty every step so robot does not waste its time
		reward -= 0.1

		# small penalty for doing nothing
		if action == 3:
			reward -= 0.05

		# penalty if robot is not facing the goal well
		reward -= 0.1 * abs(relative_angle)

		# penalty if obstacle is too close in front
		if front < 0.30:
			reward -= 1.0

		return reward

	def odom_callback(self, msg):

		# saving robot x and y position
		self.robot_x = msg.pose.pose.position.x
		self.robot_y = msg.pose.pose.position.y

		# getting quaternion orientation
		q = msg.pose.pose.orientation
		quat = [q.x, q.y, q.z, q.w]

		# quaternion conversion → roll, pitch, yaw
		roll, pitch, yaw = euler_from_quaternion(quat)

		# store yaw
		self.robot_yaw  = yaw

	def scan_callback(self, msg):
		# store lidar ranges
		self.lidar_ranges = np.array(msg.ranges, dtype=np.float32)

		# marking that a fresh scan arrived
		self.new_scan = True

	def _get_obs(self):
		# distance to goal
		# relative angle to goal
		# minimum obstacle distance in front
		# minimum obstacle distance on the left
		# minimum obstacle distance on the right


		# get current robot pose (x, y, yaw)
		# compute goal distance (dx, dy to goal)
		# compute goal angle difference
		# split LiDAR into 3 sectors
		# take min. value from each sector
		# return all 5 values as one observation vector

		# distance from robot to goal
		dx = self.goal_x - self.robot_x
		dy = self.goal_y - self.robot_y
		distance_to_goal = np.sqrt(dx**2 + dy**2)

		# angle from robot to goal
		goal_angle = np.arctan2(dy, dx)
		relative_angle = goal_angle - self.robot_yaw

		# forces angle to wrap properly, so robot always turns the shortest/most natural way toward goal
		# converts angle into shortest possible turn direction
		#wrapping angle to [-pi,pi]
		relative_angle = np.arctan2(
			np.sin(relative_angle),
			np.cos(relative_angle)
		)

		# lidar values
		if self.lidar_ranges is None:
			front = 10.0
			left = 10.0
			right = 10.0
		else:
			ranges = np.nan_to_num(
			self.lidar_ranges,
			nan = 10.0,
			posinf = 10.0,
			neginf = 10.0
			)

			n = len(ranges)
			k = 10 	#sector width

			# front sector = values near beginning and end
			front_sector = np.concatenate((ranges[:k], ranges[-k:]))

			# left sector center
			left_center = n // 4

			# right sector center
			right_center = 3 * n // 4

			left_sector = ranges[left_center - k : left_center + k]
			right_sector = ranges[right_center - k : right_center + k]

			front = np.nanmin(front_sector)
			left = np.nanmin(left_sector)
			right = np.nanmin(right_sector)

		# final observation vector
		observation = np.array([
			distance_to_goal,
			relative_angle,
			front,
			left,
			right,
		], dtype=np.float32)

		return observation

	def close(self):
		# stop robot before closing
		cmd = Twist()
		cmd.linear.x = 0.0
		cmd.angular.z = 0.0
		self.cmd_vel_pub.publish(cmd)

		# destroy ROS node
		self.node.destroy_node()

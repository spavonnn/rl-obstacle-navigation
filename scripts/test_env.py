from turtlebot_rl_env.turtlebot_gym_env import TurtlebotGymEnv
import time

# create environment
env = TurtlebotGymEnv()

# reset environment
print("\n=== Reset Test ===")
obs, info = env.reset()

print("Initial Observation:", obs)
print("Info:", info)

print("\n=== Step Test ===")

for step in range(20):

	# random action (0,1,2,3)
	action = env.action_space.sample()

	# step environment
	obs, reward, terminated, truncated, info = env.step(action)

	distance_to_goal, relative_angle, front, left, right = obs

	print(f"\n--- Step {step+1} ---")
	print("Action:", action)
	print(f"Distance to Goal: {distance_to_goal:.3f}")
	print(f"Relative Angle: {relative_angle:.3f}")
	print(f"Front: {front:.3f}, Left: {left:.3f}, Right: {right:.3f}")
	print("Reward:", reward)
	print("Terminated:", terminated)
	print("Truncated:", truncated)
	print("Info:", info)

	# small pause to watch robot in Gazebo
	time.sleep(0.2)

	# stop if episode ends
	if terminated or truncated:
		print("\n=== Episode ended ===")
		break
env.close()

import time
from stable_baselines3 import PPO
from turtlebot_gym_env import TurtlebotGymEnv

# loading the trained PPO model to run in Gazebo
# without any training — just watching the robot navigate.
# using to record the robot's behavior after training.
# Presentation Version: goals are pre-selected to demonstrate success, max steps, and collision cases.
# obstacles are still random each episode.


# pre-selected goals for presentation (odom frame)
# chosen from prior evaluation runs to show all 3 outcomes

DEMO_GOALS = [
    (-1.191, -0.42), # Episode 1: GOAL REACHED (fast, clean path)
    (-1.044, 0.302), # Episode 2: GOAL REACHED (moderate distance)
    (0.944, 1.331), # Episode 3: GOAL REACHED (longer distance)
    (-0.615, -0.791), # Episode 4: MAX STEPS (cylinders block path)
    (0.935, -0.024), # Episode 5: COLLISION (cylinders close to spawn)
]

NUM_EPISODES = len(DEMO_GOALS)

# create environment
env = TurtlebotGymEnv()

# load the trained model
print("Loading trained model from ppo_turtlebot.zip...")
model = PPO.load("ppo_turtlebot", env=env)
print("Model loaded successfully!")
print(f"Running {NUM_EPISODES} evaluation episodes...\n")

# tracking stats
episode_rewards = []
episode_lengths = []
goals_reached = 0
collisions = 0

for episode in range(NUM_EPISODES):

    # reset environment
    obs, info = env.reset()

    # override goal with pre-selected presentation goal
    # obstacles remain random, only the goal is fixed
    goal_x, goal_y = DEMO_GOALS[episode]
    env.goal_x = goal_x
    env.goal_y = goal_y

    # recompute observation with the new goal position
    obs = env._get_obs()
    env.prev_distance_to_goal = obs[0]
    
    episode_reward = 0
    episode_length = 0
    done = False

    print(f"--- Episode {episode + 1} ---")
    print(f"Goal: odom ({goal_x}, {goal_y})")

    while not done:
        # get action from trained policy (deterministic = no random exploration)
        action, _ = model.predict(obs, deterministic=True)

        # step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

    # episode finished
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

    if info.get("goal_reached"):
        goals_reached += 1
        print(f"Result: GOAL REACHED ")
    elif info.get("collision"):
        collisions += 1
        print(f"Result: COLLISION ")
    else:
        print(f"Result: MAX STEPS REACHED")

    print(f"Reward: {episode_reward:.2f}")
    print(f"Steps:  {episode_length}\n")

    # small pause between episodes so we can see the reset in Gazebo
    time.sleep(1.0)

# print summary
print("=" * 40)
print("EVALUATION SUMMARY")
print("=" * 40)
print(f"Episodes run:       {NUM_EPISODES}")
print(f"Goals reached:      {goals_reached} / {NUM_EPISODES}")
print(f"Collisions:         {collisions} / {NUM_EPISODES}")
print(f"Average reward:     {sum(episode_rewards) / NUM_EPISODES:.2f}")
print(f"Average steps:      {sum(episode_lengths) / NUM_EPISODES:.1f}")
print("=" * 40)

# close environment
env.close()
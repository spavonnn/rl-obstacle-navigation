import time
from stable_baselines3 import PPO
from turtlebot_gym_env import TurtlebotGymEnv

# loading the trained PPO model to run in Gazebo
# without any training — just watching the robot navigate.
# using to record the robot's behavior after training.


# number of episodes to run
NUM_EPISODES = 10

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
    episode_reward = 0
    episode_length = 0
    done = False

    print(f"--- Episode {episode + 1} ---")

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
        print(f"Result: GOAL REACHED ✓")
    elif info.get("collision"):
        collisions += 1
        print(f"Result: COLLISION ✗")
    else:
        print(f"Result: MAX STEPS REACHED")

    print(f"Reward: {episode_reward:.2f}")
    print(f"Steps:  {episode_length}\n")

    # small pause between episodes so you can see the reset in Gazebo
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
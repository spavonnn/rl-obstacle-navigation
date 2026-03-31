from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from turtlebot_gym_env import TurtlebotGymEnv

# create environment
env = TurtlebotGymEnv()

# check environment 
check_env(env, warn=True)

# create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_logs/",
    device="cpu"
)

# train model
model.learn(total_timesteps=50000)

# save model
model.save("ppo_turtlebot")

# close environment
env.close()
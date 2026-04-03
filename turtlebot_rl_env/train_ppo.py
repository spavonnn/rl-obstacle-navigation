import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from turtlebot_gym_env import TurtlebotGymEnv

# create environment
env = TurtlebotGymEnv()

# checkpoint callback to save model every 10k steps
# filenames: ppo_turtlebot_checkpoint_10000_steps.zip, ppo_turtlebot_checkpoint_20000_steps.zip, etc.
checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path="./ppo_checkpoints/", 
    name_prefix="ppo_turtlebot"
)

# auto resume logic - training from latest checkpoint if it exists
# if no latest checkpoint found, training starts from scratch

# it will always pick up from latest saved model. 

save_path = "ppo_turtlebot.zip"
if os.path.exists(save_path):
    print("Found existing model checkpoint. Resuming training from", save_path)
    model = PPO.load(
        save_path, 
        env=env,
        verbose=1,
        tensorboard_log="./ppo_logs/", 
        device="cpu"
    )
else:
    print("No existing model checkpoint found. Starting fresh training run...")
    # create new PPO model if no checkpoint found
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        device="cpu"
    )

# train for 50,000 steps per session
# each time script is ran it adds another 50k steps to the same model
# saves a new checkpoint every 10k steps
# total progress is saved in ppo_turtlebot.zip which is updated after every training session

steps_per_session = 50000

print(f"Training for {steps_per_session} steps this session...")
model.learn(
    total_timesteps=steps_per_session, 
    callback=checkpoint_callback,
    reset_num_timesteps=False  # keeps timestep continuous across sessions
)

# save after every session so next run can resume from here if needed
model.save("ppo_turtlebot")
print("Session complete. Model saved to ppo_turtlebot.zip")
print("Run the script again to continue training from this point for another 50,000 steps.")


# close environment
env.close()
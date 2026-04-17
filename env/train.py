from stable_baselines3 import SAC
from cfj_env import WirelessJammingEnv

# 1. Load the game board you just made
env = WirelessJammingEnv()

# 2. Initialize the SAC AI Brain
# The paper uses specific neural net sizes, which you can pass here
model = SAC("MlpPolicy", env, verbose=1)

# 3. Train the AI for 50,000 steps
print("Starting training...")
model.learn(total_timesteps=50000)

# 4. Save the trained brain to a file
model.save("sac_cfj_model")
print("Training complete and model saved!")
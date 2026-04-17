import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WirelessJammingEnv(gym.Env):
    def __init__(self):
        super(WirelessJammingEnv, self).__init__()
        
        self.num_aps = 4 # Number of Access Points
        self.num_users = 2 # Number of legitimate users
        self.num_eves = 1 # Number of eavesdroppers
        self.max_power = 1.0 # 1 Watt

        # ACTION SPACE: The AI chooses the transmit power for each AP (between 0 and 1)
        self.action_space = spaces.Box(low=0.0, high=self.max_power, shape=(self.num_aps,), dtype=np.float32)

        # OBSERVATION SPACE (State): The (x, y) coordinates of APs, Users, and Eve
        total_nodes = self.num_aps + self.num_users + self.num_eves
        self.observation_space = spaces.Box(low=0, high=50, shape=(total_nodes * 2,), dtype=np.float32)

    def reset(self, seed=None):
        # 1. Randomly drop APs, Users, and Eve onto a 50x50 grid
        # 2. Run the math to associate Users to the best APs
        # 3. Return the starting observation (the coordinates)
        self.state = np.random.uniform(low=0, high=50, size=(self.observation_space.shape[0],))
        return self.state, {}

    def step(self, action):
        # 1. Apply the powers chosen by the AI (the 'action' array)
        # 2. Calculate the Signal-to-Interference-plus-Noise-Ratio (SINR) for Users
        # 3. Calculate the SINR for Eve (this is where the path-loss math goes)
        # 4. Reward = (User Data Rate) - (Eve Data Rate)
        
        reward = 0.0 # Replace with your math
        done = True  # In this paper, each step is usually a fresh episode
        
        return self.state, reward, done, False, {}
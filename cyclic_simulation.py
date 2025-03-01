import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import Environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

if __name__ == '__main__':
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MAX_STEPS = 10000
    N_CARS = 200
    GREEN_DURATION = 40
    YELLOW_DURATION = 5

    sumoBinary = checkBinary('sumo')
    sumo_cmd = [
        sumoBinary,
        '-c', os.path.join('data', 'cfg', 'sumo_config.sumocfg'),
        '--no-step-log',
        '--waiting-time-memory', str(MAX_STEPS)
    ]

    env = Environment(sumo_cmd,
                      MAX_STEPS, N_CARS, GREEN_DURATION, YELLOW_DURATION, 80, 4)
    env.reset()

    total_rewards = []
    done = False

    while not done:
        # Get action and step environment
        next_state, reward, done = env.step(None)

        # Update state and reward
        state = next_state
        total_rewards.append(reward)

    print(f'Average reward: {np.mean(total_rewards)}')
    print(f'Total steps: {env.current_step}')
    print(f'Total arrived vehicles: {env.total_arrived_vehicles}')

    # Plot results
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Cyclic simulation')
    plt.show()

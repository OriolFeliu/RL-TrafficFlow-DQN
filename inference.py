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
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Hyperparameters
    MAX_STEPS = 5400
    N_CARS = 200

    STATE_SIZE = 4
    ACTION_SIZE = 4

    GREEN_DURATION = 40
    YELLOW_DURATION = 5

    sumoBinary = checkBinary('sumo-gui')
    sumo_cmd = [
        sumoBinary,
        '-c', os.path.join('data', 'cfg', 'sumo_config.sumocfg'),
        '--no-step-log',
        '--waiting-time-memory', str(MAX_STEPS)
    ]

    env = Environment(sumo_cmd, MAX_STEPS, N_CARS,
                      GREEN_DURATION, YELLOW_DURATION)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    agent.model.load_state_dict('dqn_model.pth')

    total_rewards = []

    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Get action and step environment
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # Update state and reward
        state = next_state
        total_rewards.append(reward)

    total_rewards.append(episode_reward)

    # Logging
    avg_reward = np.mean(total_rewards)
    print(
        f'Avg Reward: {avg_reward:.2f}, Arrived vehicles: {env.total_arrived_vehicles}, Time: {env.current_step} s.')

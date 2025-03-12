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

    print('INFERENCE SIMULATION')

    # Hyperparameters
    N_EPISODES = 50
    MAX_STEPS = 5400
    N_CARS = 200

    STATE_SIZE = 16
    ACTION_SIZE = 4

    GREEN_DURATION = 40
    YELLOW_DURATION = 5

    sumoBinary = checkBinary('sumo')
    sumo_cmd = [
        sumoBinary,
        '-c', os.path.join('data', 'cfg', 'sumo_config.sumocfg'),
        '--no-step-log',
        '--waiting-time-memory', str(MAX_STEPS)
    ]

    env = Environment(sumo_cmd, MAX_STEPS, N_CARS,
                      GREEN_DURATION, YELLOW_DURATION)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    model_path = f'dqn_{N_EPISODES}_model.pth'
    agent.load_model(model_path)

    total_rewards = []

    state = env.reset()
    done = False

    while not done:
        # Get action and step environment
        action = agent.act(state)

        next_state, reward, done = env.step(action)

        # Update state and reward
        state = next_state
        total_rewards.append(reward)


    # Logging
    avg_reward = np.mean(total_rewards)
    print(f'Average reward: {np.mean(total_rewards)}')
    print(f'Total steps: {env.current_step}')
    print(f'Total arrived vehicles: {env.total_arrived_vehicles}')

    # Plot results
    plt.plot(total_rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Trained DQN simulation')
    plt.show()
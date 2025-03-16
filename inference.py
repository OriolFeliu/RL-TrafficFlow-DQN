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
    N_EPISODES = 200
    MAX_STEPS = 5400
    N_CARS = 200

    STATE_SIZE = 16
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

    model_path = f'dqn_{N_EPISODES}_model.pth'
    agent.load_model(model_path)

    total_queue_lengths = []
    total_queue_times = []

    state = env.reset()
    done = False

    while not done:
        # Get action and step environment
        action = agent.act(state)

        next_state, reward, done = env.step(action)

        # Update state and reward
        state = next_state

        if not done:
            total_queue_lengths.append(env.get_queue_length_reward())
            total_queue_times.append(env.get_queue_waiting_time_reward())

    # Logging
    avg_queue_length = np.mean(total_queue_lengths)
    avg_queue_time = np.mean(total_queue_times)
    print(f'Average queue length: {avg_queue_length}')
    print(f'Average queue time: {avg_queue_time}')
    print(f'Total steps: {env.current_step}')
    print(f'Total arrived vehicles: {env.total_arrived_vehicles}')

    fig, ax1 = plt.subplots()

    # Plot total_queue_lengths on the primary y-axis
    ax1.plot(total_queue_lengths, label='Queue Length', color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Queue Length', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for total_queue_times
    ax2 = ax1.twinx()
    ax2.plot(total_queue_times, label='Queue Time', color='red')
    ax2.set_ylabel('Queue Time', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and grid
    plt.title('Trained DQN Simulation')
    fig.tight_layout()

    plt.show()

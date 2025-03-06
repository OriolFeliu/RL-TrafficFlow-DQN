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
    N_EPISODES = 200
    MAX_STEPS = 5400
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    GAMMA = 0.8
    LR = 1e-3
    TARGET_UPDATE = 10
    BUFFER_SIZE = 2000
    N_CARS = 200

    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.99

    STATE_SIZE = 4
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
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, EPSILON_START,
                     EPSILON_END, EPSILON_DECAY, HIDDEN_SIZE, LR, GAMMA)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    total_rewards = []
    losses = []

    for episode in range(N_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action and step environment
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)

            # Train agent
            if replay_buffer.size() > BATCH_SIZE:
                sample = replay_buffer.sample(BATCH_SIZE)
                loss = agent.train(sample)
                losses.append(loss)

            # Update state and reward
            state = next_state
            episode_reward += reward

        # Post-episode updates
        agent.update_epsilon()
        total_rewards.append(episode_reward)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target_model()

        # Logging
        avg_reward = np.mean(total_rewards[-100:])
        print(
            f'Episode: {episode+1}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Arrived vehicles: {env.total_arrived_vehicles}, Epsilon: {agent.epsilon:.2f}'
        )

    # Save model and plot results
    torch.save(agent.model.state_dict(), 'dqn_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()

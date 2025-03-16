import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import Environment
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
from config import TRAINING, ENV

if __name__ == '__main__':
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # HYPERPARAMETHERS
    # Training
    N_EPISODES = TRAINING["n_episodes"]
    MAX_STEPS = TRAINING["max_steps"]
    BATCH_SIZE = TRAINING["batch_size"]
    HIDDEN_SIZE = TRAINING["hidden_size"]
    GAMMA = TRAINING["gamma"]
    LR = TRAINING["lr"]
    TARGET_UPDATE = TRAINING["target_update"]
    BUFFER_SIZE = TRAINING["buffer_size"]
    N_CARS = TRAINING["n_cars"]
    EPSILON_START = TRAINING["epsilon_start"]
    EPSILON_END = TRAINING["epsilon_end"]
    EPSILON_DECAY = TRAINING["epsilon_decay"]

    # Environment
    STATE_SIZE = ENV["state_size"]
    ACTION_SIZE = ENV["action_size"]
    GREEN_DURATION = ENV["green_duration"]
    YELLOW_DURATION = ENV["yellow_duration"]

    N_INTERSECTIONS = 1

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
    total_losses = []

    for episode in range(N_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        episode_losses = []

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
                episode_losses.append(loss)

            # Update state and reward
            state = next_state
            episode_reward += reward

        # Post-episode updates
        agent.update_epsilon()
        total_rewards.append(np.average(episode_reward))
        total_losses.append(np.average(episode_losses))

        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target_model()

        # Logging
        avg_reward = np.mean(total_rewards[-100:])
        print(
            f'Episode: {episode+1}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Arrived vehicles: {env.total_arrived_vehicles}, Epsilon: {agent.epsilon:.2f}'
        )

    # Save model and plot results
    torch.save(agent.model.state_dict(), f'dqn_{N_EPISODES}_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label='Loss')
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

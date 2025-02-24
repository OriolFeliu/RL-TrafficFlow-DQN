import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sumolib import checkBinary

from env import Environment, TrafficGenerator
from dqn_agent import DQNAgent

if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Hyperparameters
    EPISODES = 500
    MAX_STEPS = 1000
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 10
    LR = 1e-3
    BUFFER_SIZE = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sumoBinary = checkBinary('sumo')
    sumo_cmd = [sumoBinary, "-c", os.path.join('data', 'cfg', 'sumo_config.sumocfg'), "--no-step-log", "true", "--waiting-time-memory", str(MAX_STEPS)]
    env = Environment(None, TrafficGenerator, sumo_cmd, MAX_STEPS, 10, 10, 80, 4,)
    agent = DQNAgent()

    total_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get action and step environment
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            # Store experience
            agent.buffer.push(state, action, reward, next_state, done)

            # Train agent
            agent.train()

            # Update state and reward
            state = next_state
            episode_reward += reward

        # Post-episode updates
        agent.update_epsilon()
        total_rewards.append(episode_reward)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Logging
        avg_reward = np.mean(total_rewards[-100:])
        print(
            f"Episode: {episode+1}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
        )

    # Save model and plot results
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.show()
    env.close()

# Adaptive Traffic Light Control Using Reinforcement Learning: Enhancing Urban Traffic Flow
This project aims to develop an intelligent traffic light control system using Reinforcement Learning (RL). The RL agent is trained to optimize traffic signal timings, reducing congestion and improving traffic flow efficiency. Several RL models are developed and compared using various performance metrics. The thesis and code used are available in this repository.

## Overview
This repository includes:
* Code for training and evaluating the RL agent for traffic light control.
* The thesis report explaining the methodology, experiments, and results of the project.

## Requirements
To run the code, the following libraries are required:

* Python 3.x
* TensorFlow 2.x or PyTorch (depending on the implementation)
* NumPy
* Matplotlib
* OpenAI Gym
* SUMO (Simulation of Urban Mobility)
* Pandas
* Seaborn

## Usage
The code for traffic simulation, training the RL agent, and evaluating the results can be found in the `traffic_rl_agent.ipynb` Jupyter notebook.  

1. Ensure that SUMO is installed and configured correctly on your system.  
2. Run the notebook to:
   - Simulate traffic scenarios.
   - Train the RL agent using the provided environment.
   - Evaluate and visualize the performance metrics.  

## Results
The results of the experiments are described in detail in the thesis report. Key findings include:  
* The RL agent significantly reduces average waiting time and congestion compared to traditional traffic control systems.  
* Performance improves with increased training episodes and more realistic traffic simulations.  

## Thesis
The thesis report is available in the root folder in PDF format. It provides a detailed explanation of the methodology, experimental setup, and results of the project.

## Conclusion
This project demonstrates the potential of reinforcement learning for adaptive traffic light control. The thesis discusses how the agent learns to optimize traffic flow by balancing throughput and minimizing waiting time. With further enhancements, this approach could contribute to smarter urban traffic management systems.

## References
- This project is based on the original implementation by [Andrea Vidali](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control):  
**Deep Q-Learning Agent for Traffic Signal Control**.

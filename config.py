TRAINING = {
    'n_episodes': 200,
    'max_steps': 5400,
    'batch_size': 64,
    'hidden_size': 64,
    'gamma': 0.8,
    'lr': 1e-3,
    'target_update': 10,
    'buffer_size': 2000,
    'n_cars': 200,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay': 0.99
}

ENV = {
    'state_size': 16,
    'action_size': 4,
    'green_duration': 10,
    'yellow_duration': 2
}

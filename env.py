import torch
import traci
import numpy as np
import random
import timeit
import os
import math

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Environment:
    def __init__(self, sumo_cmd, max_steps, n_cars, green_duration, yellow_duration, num_states, num_actions):
        self.n_cars = n_cars
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

        self.done = False
        self.current_step = 0
        self.total_arrived_vehicles = 0

    def reset(self):
        self.done = False
        self.current_step = 0
        self.total_arrived_vehicles = 0

        self.generate_routefile()
        traci.start(self._sumo_cmd)

        state = self.get_queue_length_state()
        return state

    def step(self, action):
        # Process action and modify traffic light state
        # traci.trafficlight.setPhase("junctionID", action)
        if action == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

        # Do a step in the simulation what will reflect the previous action
        traci.simulationStep()

        next_state = self.get_queue_length_state()
        reward = self.get_queue_length_reward()

        self.current_step += 1
        self.total_arrived_vehicles += traci.simulation.getArrivedNumber()

        # End simulation if maximum steps reached or all cars have exit the simulation
        done = (self.current_step >= self.max_steps) or (
            self.total_arrived_vehicles >= self.n_cars)
        if done:
            traci.close()

        return next_state, reward, done

    # TODO use next state to calculate this without repeating code
    def get_queue_length_reward(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")

        queue_length = halt_N + halt_S + halt_E + halt_W

        return -queue_length

    def get_queue_length_state(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")

        return np.array([halt_N, halt_S, halt_E, halt_W,])

    def generate_routefile(self):
        shape_param = 2
        timings = np.random.weibull(shape_param, self.n_cars)
        timings = np.sort(timings)

        if self.n_cars == 0:
            return np.array([])

        # Find actual min/max of generated data (not using floor/ceiling)
        min_old = timings[0]  # Use first element, not floor(timings[1])
        max_old = timings[-1]

        # Handle edge case where all values are identical
        if math.isclose(min_old, max_old, rel_tol=1e-9):
            return np.full(self.n_cars, self.max_steps // 2)

        # Linear transformation to [0, max_steps] using vectorization
        car_gen_steps = (timings - min_old) * \
            (self.max_steps / (max_old - min_old))

        # Round to integers (use floor() to avoid exceeding max_steps)
        car_gen_steps = np.floor(car_gen_steps).astype(int)

        # Clip to ensure values stay within [0, max_steps-1]
        car_gen_steps = np.clip(car_gen_steps, 0, self.max_steps-1)
        # car_gen_steps = np.clip(car_gen_steps, 0, self.max_steps*0.75)

        with open("data/route/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

        # produce the file for cars generation, one car per line
        with open("data/route/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    # choose a random source & destination
                    route_straight = np.random.randint(1, 5)
                    if route_straight == 1:
                        print('    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    # choose random source source & destination
                    route_turn = np.random.randint(1, 9)
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' %
                              (car_counter, step), file=routes)

            print("</routes>", file=routes)

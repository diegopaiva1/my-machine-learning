import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from collections import deque

from blob import Blob

# Hyperparameters for the DQN
MIN_REPLAY_MEMORY_SIZE = 10_000
MAX_REPLAY_MEMORY_SIZE = 50_000
UPDATE_TARGET_EVERY = 5
MINI_BATCH_SIZE = 64
DISCOUNT = 0.95

class Agent(Blob):
    def __init__(self, env):
        super().__init__(env)

        # Main model, gets trained every step
        self.model = self.create_model()

        # Target model, this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE)

        # Used to count when to update target model with main model's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), activation = 'relu', input_shape = self.env.observation_space))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.2))

        # Hidden layer
        model.add(Flatten())
        model.add(Dense(64, activation = 'relu'))
        model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(self.env.get_action_space_size(), activation = 'linear'))

        model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

        return model

    def color(self):
        return (255, 0, 0)

    def get_q_values(self, observation):
        return self.model.predict(np.array(observation).reshape(-1, *observation.shape)/255)[0]

    def train(self, done):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        current_observations = np.array([transition.current_observation for transition in mini_batch])/255
        current_q_values = self.model.predict(current_observations)

        new_observations = np.array([transition.new_observation for transition in mini_batch])/255
        future_q_values = self.target_model.predict(new_observations)

        X = []
        y = []

        for i, (current_observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                # Bellman equation
                new_q = reward + DISCOUNT * np.max(future_q_values[i])
            else:
                new_q = reward

            current_q_values[i][self.env.action_space.index(action)] = new_q

            X.append(current_observation)
            y.append(current_q_values)

        self.model.fit(np.array(X)/255, np.array(y), batch_size = MINI_BATCH_SIZE, verbose = 0, shuffle = False)

        # Update target model counter every episode
        if done:
            self.target_update_counter += 1

        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

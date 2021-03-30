'''
The architecture of a DQN is like a standard neural network where we pass an input.
It produces an output and then we take the loss let’s say (predicted_value - actual_value) and then we update the weights.
For the input, because we are using images, we use a CNN and then pass the results to a sequential model to produce some output value for each action.
The major difference is calculating the loss, this is where replay memory and target network comes into play.
Because RL is not like supervised learning where we have clear labels for our data, we use the replay memory and the target network to come up
the “actual value” that we then use for the loss (predicted_value – actual_value).
This means that before we start training our network, we need to have the actual values for the states, so how do we get them?

Replay Memory:
Remember that the network outputs q values that corresponds to an action that the agent can take.
We pass in our state, it produces different q values we then pick one of those q values using epsilon greedy and take the action that is associated with that q value.
After we have taken that action in the environment, we get back a reward and a new state.
So, we now have a tuple that consists of (s_t, a_t, r_{t+1}, s_{t+1}) We then append that tuple to a list.
We keep repeating this process until we reached some desired list size.

Target Network:
Now that we have the replay memory-filled, we can start training.
We go and randomly pick a tuple from our replay memory. We pass in the state (s_t) that is in the tuple to the network and it produces some q values.
We then pick the q value that is associated with the action that we previously took in the tuple (a_t). This q value is going to be our “Predicted_value”.
To get the “actual_value” we pass in the s_{t+1} that is in our tuple to another network called the Target Network.
This network has the same architecture as our model, the only difference is that the weights remain frozen but every so often, we update the weights in
the target network to the same weights that we have in our model.
The reason we use another network instead of our model for this step is that we need to make sure the weights don’t go changing every time.
If they change then every time, we pass in s_{t+1} we would get a different “actual_value” for the same (state, action) pair so we wouldn’t be able to minimize the loss.
So, freezing the values makes sure that this does not happen and that all the different states get treated the same way.

So anyway, we pass in s_{t+1} to the target network and then get the max q value that it produces.
After that we pick that q value and the r_{t+1} in our tuple and we pass it into the Bellman equation.
[r_{t+1} + discount (q value)] this gives us the “actual_value”.

After that just plug it into the loss.
Loss: (predicted_value – actual_value) and then just do normal backpropagation.
'''

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

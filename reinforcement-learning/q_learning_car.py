import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
RENDER_EVERY = 2500
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

env = gym.make('MountainCar-v0')

def get_discrete_observation(observation):
    discrete_observation = (observation - env.observation_space.low)/batch_size
    return tuple(discrete_observation.astype(np.int))

discrete_observation_space_size = [20] * len(env.observation_space.high)
batch_size = (env.observation_space.high - env.observation_space.low)/discrete_observation_space_size
q_table = np.random.uniform(low = -2, high = 0, size = discrete_observation_space_size + [env.action_space.n])

# Give our agent the opportunity to conduct a random exploration
epsilon = 0.5
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

episode_rewards = []
aggr_episode_rewards = {'episodes': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES + 1):
    if episode % RENDER_EVERY == 0:
        render = True
    else:
        render = False

    current_discrete_observation = get_discrete_observation(env.reset())
    episode_reward = 0
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[current_discrete_observation])
        else:
            action = env.action_space.sample()

        new_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_observation = get_discrete_observation(new_observation)

        if render:
            env.render()

        if not done:
            current_q = q_table[current_discrete_observation + (action, )]
            max_future_q = np.max(q_table[new_discrete_observation])
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update q-value for the taken action
            q_table[current_discrete_observation + (action, )] = new_q
        elif new_observation[0] >= env.goal_position:
            q_table[current_discrete_observation + (action, )] = 0

        current_discrete_observation = new_discrete_observation

    if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    episode_rewards.append(episode_reward)

    if episode % RENDER_EVERY == 0:
        avg_reward = sum(episode_rewards[-RENDER_EVERY:])/len(episode_rewards[-RENDER_EVERY:])
        min_reward = min(episode_rewards[-RENDER_EVERY:])
        max_reward = max(episode_rewards[-RENDER_EVERY:])

        aggr_episode_rewards['episodes'].append(episode)
        aggr_episode_rewards['avg'].append(avg_reward)
        aggr_episode_rewards['min'].append(min_reward)
        aggr_episode_rewards['max'].append(max_reward)

        print(f'Episode {episode}/{EPISODES}, avg: {avg_reward:.3f}, min: {min_reward}, max: {max_reward} ε: {epsilon:.3f}', end = '\r', flush = True)

env.close()

fig, ax = plt.subplots()
ax.set_title('Agent analysis')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.plot(aggr_episode_rewards['episodes'], aggr_episode_rewards['avg'], label = 'avg')
ax.plot(aggr_episode_rewards['episodes'], aggr_episode_rewards['min'], label = 'min')
ax.plot(aggr_episode_rewards['episodes'], aggr_episode_rewards['max'], label = 'max')
ax.legend()
fig.savefig('episode_vs_reward.png')

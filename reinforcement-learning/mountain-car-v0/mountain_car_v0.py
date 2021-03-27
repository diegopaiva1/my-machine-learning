import gym
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 50_000
RENDER_EVERY = 10_000
EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

def main():
    env = gym.make('MountainCar-v0')

    discrete_observation_space_size = [20] * len(env.observation_space.high) # or .low, does not really matter
    batch_size = (env.observation_space.high - env.observation_space.low)/discrete_observation_space_size

    # initializing q-table with random values with proper dimension
    q_table = np.random.uniform(low = -2, high = 0, size = discrete_observation_space_size + [env.action_space.n])

    # give our agent the opportunity to conduct a random exploration
    epsilon = EPSILON
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    cum_rewards = 0
    cum_avg_rewards = []

    for episode in range(EPISODES):
        current_discrete_observation = get_discrete_observation(env, env.reset(), batch_size)
        total_episode_reward = 0
        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[current_discrete_observation])
            else:
                action = env.action_space.sample()

            new_observation, reward, done, _ = env.step(action)
            total_episode_reward += reward
            new_discrete_observation = get_discrete_observation(env, new_observation, batch_size)

            if episode % RENDER_EVERY == 0:
                env.render()

            if not done:
                current_q = q_table[current_discrete_observation + (action, )]
                max_future_q = np.max(q_table[new_discrete_observation])
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # update q-value for the taken action
                q_table[current_discrete_observation + (action, )] = new_q
            elif new_observation[0] >= env.goal_position:
                # well done
                q_table[current_discrete_observation + (action, )] = 0

            current_discrete_observation = new_discrete_observation

        if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        cum_rewards += total_episode_reward
        avg_reward = cum_rewards/(episode + 1)
        cum_avg_rewards.append(avg_reward)

        print(f'Episode {episode}/{EPISODES}, avg: {avg_reward:.3f}, ε: {epsilon:.3f}', end = '\r', flush = True)

    plot_episode_vs_reward(cum_avg_rewards)

    env.close()

def get_discrete_observation(env, observation, batch_size):
    ''' Get the observation as discrete values.

    Parameters
    ----------
    `env`: the environment.
    `observation`: original observation values.
    `batch_size`: batch size of the discrete observation space.

    Returns
    -------
    The discrete observation in tuple form.
    '''

    discrete_observation = (observation - env.observation_space.low)/batch_size
    return tuple(discrete_observation.astype(np.int))

def plot_episode_vs_reward(cum_avg_rewards, figname = 'episode_vs_reward.png'):
    ''' Plot the episodes against their cumulative average rewards.

    Parameters
    ----------
    `cum_avg_rewards`: list containing the cumulative average rewards per episode.
    `figname`: name of the figure to be saved, default: 'episode_vs_reward.png'.
    '''

    fig, ax = plt.subplots()

    ax.set_title(f'Cumulative average reward per episode (ε = {EPSILON})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.plot([episode for episode in range(len(cum_avg_rewards))], cum_avg_rewards)

    fig.savefig(figname)

if __name__ == '__main__':
    main()

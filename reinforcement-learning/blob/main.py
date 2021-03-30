import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple

# Our custom environment
from env import Env

plt.style.use('ggplot')

# Hyperparameters
EPISODES = 100
RENDER_EVERY = 1
EPSILON = 0.50
MAX_STEPS_PER_EPISODE = 200
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def main():
    env = Env(width = 10, height = 10)

    # Give the agent the opportunity to conduct a random exploration
    epsilon = EPSILON

    # Let us define a data structure for registering the transitions in the environment
    Transition = namedtuple('Transition', ['current_observation', 'action', 'reward', 'new_observation', 'done'])

    cum_reward = 0
    cum_avg_rewards = []

    for episode in range(EPISODES):
        current_observation = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if random.random() > epsilon:
                action = env.action_space[np.argmax(env.agent.get_q_values(current_observation))]
            else:
                action = random.choice(env.action_space)

            new_observation, reward, done = env.step(action)
            episode_reward += reward
            episode_steps += 1

            if episode % RENDER_EVERY == 0:
                env.render()

            env.agent.replay_memory.append(Transition(current_observation, action, reward, new_observation, done))
            env.agent.train(done)

            current_observation = new_observation

            if episode_steps == MAX_STEPS_PER_EPISODE:
                break

        if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
            epsilon -= EPSILON_DECAY_VALUE

        cum_reward += episode_reward
        avg_reward = cum_reward/(episode + 1)
        cum_avg_rewards.append(avg_reward)

        print(f'Episode {episode}/{EPISODES}, avg: {avg_reward:.3f}, ε: {epsilon:.3f}', end = '\r', flush = True)

    env.agent.model.save('models/2-256conv-1-64dense')

    plot_episode_vs_reward(cum_avg_rewards)

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

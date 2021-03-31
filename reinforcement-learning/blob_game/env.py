import numpy as np
import random
import cv2
import imutils
from PIL import Image

from agent import Agent
from food import Food
from enemy import Enemy

class Env:
    def __init__(self, width = 10, height = 10, enemy_reward = -300, move_reward = -1, food_reward = 25):
        self.observation_space = (width, height, 3) # w x h BGR image
        self.action_space = [
            (1, 0),   # East
            (-1, 0),  # West
            (0, 1),   # North
            (0, -1),  # South
            (1, 1),   # Northeast
            (-1, 1),  # Northwest
            (1, -1),  # Southeast
            (-1, -1)  # Southwest
        ]

        # Grid dimension
        self.width = width
        self.height = height

        # Custom reward parameters
        self.enemy_reward = enemy_reward
        self.move_reward = move_reward
        self.food_reward = food_reward

    def reset(self):
        # Unlike food and enemy, the agent cannot be reinitialized, so as to preserve its replay memory
        if hasattr(self, 'agent'):
            self.agent.x = random.randint(0, self.width - 1)
            self.agent.y = random.randint(0, self.height - 1)
        else:
            self.agent = Agent(self)

        self.food = Food(self)

        while self.food.same_cell(self.agent):
            self.food = Food(self)

        self.enemy = Enemy(self)

        while self.enemy.same_cell(self.agent) or self.enemy.same_cell(self.food):
            self.enemy = Enemy(self)

        return self.get_image()

    def step(self, action, move_enemy = False, move_food = False):
        self.agent.move(action)

        if move_enemy:
            self.enemy.move(random.choice(self.action_space))

        if move_food:
            self.food.move(random.choice(self.action_space))

        new_observation = self.get_image()

        if self.agent.same_cell(self.enemy):
            reward = self.enemy_reward
            done = True
        elif self.agent.same_cell(self.food):
            reward = self.food_reward
            done = True
        else:
            reward = self.move_reward
            done = False

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = imutils.resize(img, width = 300)

        cv2.imshow('env', img)
        cv2.waitKey(1)

    def get_image(self):
        # Starts a BGR of observation space dimension
        env = np.zeros(self.observation_space, dtype = np.uint8)

        env[self.agent.x][self.agent.y] = self.agent.color()
        env[self.food.x][self.food.y] = self.food.color()
        env[self.enemy.x][self.enemy.y] = self.enemy.color()

        return np.array(Image.fromarray(env))

    def get_action_space_size(self):
        return len(self.action_space)

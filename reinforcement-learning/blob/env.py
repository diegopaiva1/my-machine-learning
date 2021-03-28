import numpy as np
import cv2
import imutils
from PIL import Image

from agent import Agent
from food import Food
from enemy import Enemy

class Env:
    def __init__(self):
        self.width = 10
        self.height = 10

        # Let's create our blobs is unique positions
        self.agent = Agent(self.width, self.height)
        self.food = Food(self.width, self.height)

        while self.food.same_cell(self.agent):
            self.food = Food(self.width, self.height)

        self.enemy = Enemy(self.width, self.height)

        while self.enemy.same_cell(self.agent) or self.enemy.same_cell(self.food):
            self.enemy = Enemy(self.width, self.height)

        print(f'Agent spawn: {self.agent.x, self.agent.y}')
        print(f'Food spawn: {self.food.x, self.food.y}')
        print(f'Enemy spawn: {self.enemy.x, self.enemy.y}')

    def render(self):
        img = self.get_image()
        img = imutils.resize(img, width = 300)

        cv2.imshow('env', img)
        cv2.waitKey()

    def get_image(self):
        # Starts an BGR of our size
        env = np.zeros((self.width, self.height, 3), dtype = np.uint8)

        env[self.agent.x][self.agent.y] = self.agent.color()
        env[self.food.x][self.food.y] = self.food.color()
        env[self.enemy.x][self.enemy.y] = self.enemy.color()

        return np.array(Image.fromarray(env))

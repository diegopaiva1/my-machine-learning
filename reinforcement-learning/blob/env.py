import numpy as np
import cv2
import imutils
from PIL import Image
from blob import Blob

class Env:
    def __init__(self):
        self.width = 10
        self.height = 10

        # Let's create our blobs is unique positions
        self.agent = Blob(self)
        self.food = Blob(self)

        while self.food.same_cell(self.agent):
            self.food = Blob(self)

        self.enemy = Blob(self)

        while self.enemy.same_cell(self.agent) or self.enemy.same_cell(self.food):
            self.enemy = Blob(self)

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

        # Set blob tiles colors
        env[self.agent.x][self.agent.y] = (255, 0, 0)
        env[self.food.x][self.food.y] = (0, 255, 0)
        env[self.enemy.x][self.enemy.y] = (0, 0, 255)

        return np.array(Image.fromarray(env))

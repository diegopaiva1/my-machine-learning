import random
from abc import ABC, abstractmethod

class Blob(ABC):
    def __init__(self, env):
        self.env = env
        self.x = random.randint(0, self.env.width - 1)
        self.y = random.randint(0, self.env.height - 1)

    def move(self, action):
        if action not in self.env.action_space:
            raise ValueError(f'Invalid action \'{action}\'')

        x_increment, y_increment = action

        self.x += x_increment
        self.y += y_increment

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.env.width - 1:
            self.x = self.env.width - 1

        if self.y < 0:
            self.y = 0
        elif self.y > self.env.height - 1:
            self.y = self.env.height - 1

    def same_cell(self, other):
        return self.x == other.x and self.y == other.y

    @abstractmethod
    def color(self):
        pass

from blob import Blob

class Food(Blob):
    def __init__(self, env):
        super().__init__(env)

    def color(self):
        return (0, 255, 0)

from blob import Blob

class Enemy(Blob):
    def __init__(self, env):
        super().__init__(env)

    def color(self):
        return (0, 0, 255)

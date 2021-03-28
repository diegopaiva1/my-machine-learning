from blob import Blob

class Agent(Blob):
    def __init__(self, width, height):
        super().__init__(width, height)

    def color(self):
        return (255, 0, 0)

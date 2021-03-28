from blob import Blob

class Enemy(Blob):
    def __init__(self, width, height):
        super().__init__(width, height)

    def color(self):
        return (0, 0, 255)

class AverageBuilder:

    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def add(self, x: float):
        self.n += 1
        self.sum += x

    def average(self):
        return self.sum / float(self.n)

    def clear(self):
        self.n = 0
        self.sum = 0.0

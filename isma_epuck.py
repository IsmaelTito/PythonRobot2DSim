from Robots import Epuck
import numpy as np


class IsmaEpuck(Epuck):
    def __init__(self, position=(0, 0), angle=np.pi / 2, r=0.48, bHorizontal=False, frontIR=2, nother=0, nrewsensors=0):
        Epuck.__init__(self, position, angle, r, bHorizontal, frontIR, nother, nrewsensors)

    @property
    def left_wheel(self):
        return self.motors[0]

    @left_wheel.setter
    def left_wheel(self, value):
        self.motors[0] = value

    @property
    def right_wheel(self):
        return self.motors[1]

    @right_wheel.setter
    def right_wheel(self, value):
        self.motors[1] = value

    def prox_activations(self):
        return 1.0 - np.array(self.body.userData["IRValues"])

    # def update(self):
        # Epuck.update()
        # left, right = self.prox_activations()
        # self.left_wheel = 1.0 - left   # to check
        # self.right_wheel = 1.0 - right   # to check

from Robots import Epuck
import numpy as np
from Box2DWorld import vrotate
from VectorFigUtils import vnorm, dist

class IsmaEpuck(Epuck):
    def __init__(self, position=(0, 0), angle=np.pi / 2, r=0.48, bHorizontal=False, frontIR=2, nother=0, nrewsensors=0):
        Epuck.__init__(self, position, angle, r, bHorizontal, frontIR, nother, nrewsensors)
        self.motor_commands = []
        self.avoid_walls_w = 0.9
        self.avoid_epuck_w = 0.5
        self.seek_reward_w = 0.9

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

    def epuck_sensors(self):
        return 1.0 - np.array(self.body.userData["OtherValues"])

    def reward_sensors(self):
        return 1.0 - np.array(self.body.userData["RewardValues"])

    def avoid_walls(self):
        right, left = self.prox_activations()  # I reverse this because the first value of the array corresponds to the right sensor
        left_wheel = 1.0 - right   # to check
        right_wheel = 1.0 - left   # to check
        self.add_forces(values=[left_wheel, right_wheel])
        self.avoid_walls_w = max(left, right)
        print "highest sensor value:", self.avoid_walls_w
        print "wall sensors: ", left, right
        print "wall motors: ", left_wheel, right_wheel
        # print(left, right)

    def avoid_epuck(self):
        left, right = self.epuck_sensors()
        left_wheel = 1.0 - left   # to check
        right_wheel = 1.0 - right   # to check
        self.add_forces(values=[left_wheel, right_wheel])
        # print(left, right)

    def seek_reward(self):
        left_lreward, left_hreward, right_hreward, right_lreward = self.reward_sensors()
        left_wheel = 1.0 - right_lreward
        right_wheel = 1.0 - left_lreward
        self.add_forces(values=[left_wheel, right_wheel])
        # self.left_wheel = 1.0 - right_hreward
        # self.right_wheel = 1.0 - left_hreward
        # print(left_hreward, right_hreward)

    def add_forces(self, values=[0, 0]):
        self.motor_commands.append(values)

    def apply_forces(self):
        print "motor commands: ", self.motor_commands
        print "motor commands length: ", len(self.motor_commands)
        final_values = np.sum(self.motor_commands, axis=0)
        print "final values: ", final_values
        final_values_norm = np.divide(final_values, len(self.motor_commands))
        print "final values normalized: ", final_values_norm
        print "final values normalized left: ", final_values_norm[0]
        print "final values normalized right: ", final_values_norm[1]
        # u = (2, 1)
        # final_values_vnorm = vnorm(u)
        # print "final values normalized vnorm: ", final_values_vnorm
        self.left_wheel = final_values_norm[0]
        self.right_wheel = final_values_norm[1]
        self.motor_commands = []

    def update(self):
        """update of position applying forces and IR."""
        body, angle, pos = self.body, self.body.angle, self.body.position
        mLeft, mRight = self.motors
        fangle, fdist = 50 * (mRight - mLeft), 1000 * (mLeft + mRight)
        d = (fdist * np.cos(angle), fdist * np.sin(angle))

        if(self.bHorizontal):
            d = vrotate(d, np.pi / 2)

        if(self.bForceMotors):
            body.ApplyTorque(fangle, wake=True)
            body.ApplyForce(force=d, point=body.worldCenter, wake=False)

        if(self.bHorizontal):
            body.angularVelocity = 0
            body.angle = np.pi / 2

        nir = self.frontIR
        self.IR.update(pos, angle, self.r)

        self.avoid_walls()
        self.avoid_epuck()
        self.seek_reward()
        self.apply_forces()

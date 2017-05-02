from Robots import Epuck
import numpy as np
from Box2DWorld import vrotate
from VectorFigUtils import vnorm, dist
from numpy.random import randn
from numpy import arange, exp
from config import config_data


class IsmaEpuck(Epuck):
    def __init__(self, position=(0, 0), angle=np.pi / 2, r=0.48, bHorizontal=False, frontIR=2, nother=0, nrewsensors=0):
        Epuck.__init__(self, position, angle, r, bHorizontal, frontIR, nother, nrewsensors)
        self.motor_commands = []
        # self.avoid_walls_w = 0.9
        # self.avoid_epuck_w = 0.5
        # self.seek_high_reward_w = 0.9
        # self.seek_low_reward_w = 0.5
        # self.reward_score = 0
        # print "Epuck initial reward score: ", self.reward_score

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
        w_sensors = self.prox_activations()
        dim_w_sensors = len(self.prox_activations())
        sigma = 0.01
        # add some random noise to the sensors
        w_sensors += sigma * randn(dim_w_sensors)
        right, left = w_sensors  # I reverse this because the first value of the array corresponds to the right sensor
        # map the sensors' data exponentially to the motor commands
        left_exp = exp(2*left)
        right_exp = exp(2*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0.3
        left_wheel = fwd + left_exp - right_exp  # to check
        right_wheel = fwd + right_exp - left_exp   # to check
        self.add_forces(values=[left_wheel, right_wheel])
        self.avoid_walls_w = max(left, right)  # weight value, not yet implemented
        # print "highest sensor value:", self.avoid_walls_w
        # print "wall sensors: ", left, right
        # print "wall motors: ", left_wheel, right_wheel

    def avoid_epuck(self):
        e_sensors = self.epuck_sensors()
        dim_e_sensors = len(self.epuck_sensors())
        sigma = config_data['epuck_error']  # around 0.05
        # print "avoid_epuck error:", sigma
        # add some random noise to the sensors
        e_sensors += sigma * randn(dim_e_sensors)
        right, left = e_sensors  # I reverse this because the first value of the array corresponds to the right sensor
        # map the sensors' data exponentially to the motor commands
        left_exp = exp(config_data['epuck_exp']*left)
        right_exp = exp(config_data['epuck_exp']*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0
        left_wheel = fwd + left_exp - right_exp  # to check
        right_wheel = fwd + right_exp - left_exp   # to check
        self.add_forces(values=[left_wheel, right_wheel])
        self.avoid_epuck_w = max(left, right)  # weight value, not yet implemented
        # print(left, right)

    def seek_rewards(self):
        r_sensors = self.reward_sensors()
        dim_r_sensors = len(self.reward_sensors())
        sigma = config_data['reward_error']
        # add some random noise to the sensors
        r_sensors += sigma * randn(dim_r_sensors)
        right_lreward, right_hreward, left_hreward, left_lreward = r_sensors
        self.seek_high_reward(left_hreward, right_hreward)
        self.seek_low_reward(left_lreward, right_lreward)

    def seek_high_reward(self, left=0, right=0):
        # map the sensors' data exponentially to the motor commands
        left_exp = exp(config_data['reward_exp']*left)
        right_exp = exp(config_data['reward_exp']*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0.3
        left_wheel = fwd + right_exp - left_exp
        right_wheel = fwd + left_exp - right_exp
        # left_wheel = np.exp(2 * (1 - left)) - 1
        # right_wheel = np.exp(2 * (1 - right)) - 1
        # print "Left sensor value", left
        # print "Right sensor value", right
        self.add_forces(values=[left_wheel, right_wheel])
        # check if the epuck arrived at the low-reward spot
        # if left >= 0.9 and left < 1 or right >= 0.9 and right < 1:
        #     self.reward_score += 2
        #     print "High Reward obtained!! ", self.reward_score

    def seek_low_reward(self, left=0, right=0):
        # map the sensors' data exponentially to the motor commands
        left_exp = exp(config_data['reward_exp']*left)
        right_exp = exp(config_data['reward_exp']*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0
        left_wheel = fwd + right_exp - left_exp
        right_wheel = fwd + left_exp - right_exp
        self.add_forces(values=[left_wheel * config_data['l_reward_weight'], right_wheel * config_data['l_reward_weight']])
        # check if the epuck arrived at the low-reward spot
        # if left >= 0.9 and left < 1 or right >= 0.9 and right < 1:
        #     self.reward_score += 1
        #     print "Low Reward obtained!! ", self.reward_score

    def add_forces(self, values=[0, 0]):
        self.motor_commands.append(values)

    def apply_forces(self):
        # print "motor commands: ", self.motor_commands
        # print "motor commands length: ", len(self.motor_commands)
        final_values = np.sum(self.motor_commands, axis=0)
        # print "final values: ", final_values
        final_values_avg = np.divide(final_values, len(self.motor_commands))
        # print "final values average: ", final_values_avg
        # print "final values average left: ", final_values_avg[0]
        # print "final values average right: ", final_values_avg[1]
        self.left_wheel = final_values_avg[0]
        self.right_wheel = final_values_avg[1]
        self.motor_commands = []

    def update(self):
        # self.avoid_walls()
        self.avoid_epuck()
        self.seek_rewards()
        self.apply_forces()

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

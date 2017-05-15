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
        self.high_reward_value = 1
        self.avoid_epuck_w = 0.5
        # self.seek_high_reward_w = 1.0
        # self.seek_low_reward_w = 0.5

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

    def avoid_epuck(self):
    	# get the sensors' data
        e_sensors = self.epuck_sensors()
        # add some random noise to the sensors' data
        dim_e_sensors = len(self.epuck_sensors())
        sigma = config_data['epuck_error']  # initially around 0.05
        e_sensors += sigma * randn(dim_e_sensors)
        # store sensor data in their corresponding variables 
        right, left = e_sensors  # I reverse this because the first value of the array corresponds to the right sensor
        # map the sensors' data exponentially to the motor commands
        left_exp = exp(config_data['epuck_exp']*left)
        right_exp = exp(config_data['epuck_exp']*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0.3
        left_wheel = fwd + left_exp - right_exp  # to check
        right_wheel = fwd + right_exp - left_exp   # to check
        # send the resulting motor commands to a command-integration function
        # self.add_forces(values=[left_wheel, right_wheel])
        # self.avoid_epuck_w = max(left, right)  # weight value
        # print "Avoid Epuck Weight:", self.avoid_epuck_w
        self.add_forces(values=[left_wheel * self.avoid_epuck_w, right_wheel * self.avoid_epuck_w])

    def seek_rewards(self):
    	# get the sensors' data
        r_sensors = self.reward_sensors()
        # add some random noise to the sensors' data
        dim_r_sensors = len(self.reward_sensors())
        sigma = config_data['reward_error'] # initially around 0.05
        r_sensors += sigma * randn(dim_r_sensors)
        # store sensor data in their corresponding variables 
        right_lreward, right_hreward, left_hreward, left_lreward = r_sensors
        # send the sensory information to the corresponding reactive-behavior function
        self.seek_high_reward(left_hreward, right_hreward)
        self.seek_low_reward(left_lreward, right_lreward)

    def seek_high_reward(self, left=0, right=0):
    	c = 0.8
    	# map the sensors' data exponentially to the motor commands
        left_exp = c * self.high_reward_value * exp(config_data['reward_exp']*left)
        right_exp = c * self.high_reward_value * exp(config_data['reward_exp']*right)
        # left_exp = exp(config_data['reward_exp']*left)
        # right_exp = exp(config_data['reward_exp']*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0.3
        left_wheel = fwd + right_exp - left_exp
        right_wheel = fwd + left_exp - right_exp
        # send the resulting motor commands to a command-integration function
        self.add_forces(values=[left_wheel, right_wheel])

    def seek_low_reward(self, left=0, right=0):
    	c = 0.8
        # map the sensors' data exponentially to the motor commands
        left_exp = c * exp(config_data['reward_exp']*left)
        right_exp = c * exp(config_data['reward_exp']*right)
        # left_exp = exp(config_data['reward_exp']*left)
        # right_exp = exp(config_data['reward_exp']*right)
        # link sensor data with motor activations a la Braitenberg
        fwd = 0.3
        left_wheel = fwd + right_exp - left_exp
        right_wheel = fwd + left_exp - right_exp
        # send the resulting motor commands to a command-integration function
        self.add_forces(values=[left_wheel, right_wheel])
        # self.add_forces(values=[left_wheel * config_data['l_reward_weight'], right_wheel * config_data['l_reward_weight']])

    def add_forces(self, values=[0, 0]):
    	# receive motor commands and store them together
        self.motor_commands.append(values)

    def apply_forces(self):
    	# sum all the motor commands
        final_values = np.sum(self.motor_commands, axis=0)
        # divide the result of the sum by the number of total motor commands to get an averaged final value
        final_values_avg = np.divide(final_values, len(self.motor_commands))
        # send the final values to each motor
        self.left_wheel = final_values_avg[0]
        self.right_wheel = final_values_avg[1]
        # reset the motor command storage
        self.motor_commands = []

    def update(self):
        # update the reactive functions
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

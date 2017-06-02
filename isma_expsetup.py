import numpy as np
import Box2D
import csv
import string
import random
from Box2DWorld import (world, createBox, createCircle, collisions)
from VectorFigUtils import vnorm, dist
from ExpRobotSetup import ExpSetupEpuck
from isma_epuck import IsmaEpuck
from numpy.random import rand
from config import config_data


def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for i in range(length))


def addWalls(pos, dx=3, dh=0, h=2.8, th=0, bHoriz=True, bVert=True):
    x, y = pos
    wl = 0.2
    yh = (5 + 1) / 2.0
    if(bVert):
        # createBox((x, y - 1 - dh + th), w=h + dh + wl + th, h=wl, bDynamic=False, name="wall_top")
        createBox((x, y - 0.95 - dh + th), w=h + dh + wl + th + 0.8, h=wl, bDynamic=False, name="wall_bottom")  # horizontal abajo
        # createBox((x, y + 5 + dh + th), w=h + dh + wl + th, h=wl, bDynamic=False, name="wall_bottom")
        createBox((x, y + 6.05 + dh + th), w=h + dh + wl + th + 0.8, h=wl, bDynamic=False, name="xwall_top")  # horizontal arriba
    if(bHoriz):
        # createBox((x - dx - wl, y + yh - 1 + dh / 2 + th), w=wl, h=h + dh, bDynamic=False, name="wall_left")
        createBox((x - dx - wl - 0.1, y + yh - 0.85 + dh / 2 + th), w=wl, h=h + 0.3 + dh, bDynamic=False, name="wall_left")  # vertical izq
        # createBox((x + dx + wl, y + yh - 1 + dh / 2 + th), w=wl, h=h + dh, bDynamic=False, name="wall_right")
        createBox((x + dx + wl + 0.1, y + yh - 0.85 + dh / 2 + th), w=wl, h=h + 0.3 + dh, bDynamic=False, name="wall_right")  # vertical der


def addReward(who, pos=(0, 0), vel=(0, 0), reward_type=0, bDynamic=True, bCollideNoOne=False):
    if(reward_type == 0):
        name, r = "reward", 0.27
    else:
        name, r = "reward_small", 0.2

    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=10, name=name, r=r)
    obj.userData["energy"] = 1.0
    obj.userData["visible"] = 1.0
    obj.linearVelocity = vel
    who.objs.append(obj)


def addCircle(who, pos=(0, 0), vel=(0, 0), bDynamic=False, bCollideNoOne=False):
    r = 1.06  # PREVIOUSLY WAS 0.75
    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=0, r=r)
    obj.userData["energy"] = 1.0
    obj.userData["visible"] = 1.0
    obj.linearVelocity = vel
    # who.objs.append(obj)


class IsmaExpSetup(object):
    """Exp setup class with two epucks and two reward sites."""

    def __init__(self, n=1, rounds=50, payoff="high", contextual=False, debug=False):
        """Initialize the variables we need to store in each round of the game"""
        self.parameters = [config_data['epuck_error'], config_data['reward_error'], config_data['epuck_exp'], config_data['reward_exp'], config_data['l_reward_weight'], config_data['reward_area'], config_data['reward_c']]
        self.dyadID = "game_"+id_generator(8)+"-"+id_generator(4)+"-"+id_generator(4)+"-"+id_generator(4)+"-"+id_generator(12)
        self.payoff_structure = payoff         # HIGH: 4v1, LOW: 2v1
        self.contextual = contextual
        self.total_rounds = rounds
        self.round_n = 1
        self.round_data = []
        self.timer, self.timestep, self.timeout_n = 0, 0, 0
        self.timeout = False
        self.high_reward_location = "-"
        self.player1, self.player2 = "host", "other"
        self.player1_pos, self.player2_pos = (0, 0), (0, 0)
        self.player1_ang, self.player2_ang = 0, 0
        self.player1_score, self.player2_score = 0, 0
        self.player1_wins, self.player2_wins, self.ties_n = 0, 0, 0

        """Create the two epucks, two rewards with circles around and walls."""
        global bDebug
        bDebug = debug
        print "-------------------------------------------------"
        print "Initial parameters:", self.parameters
        print "Dyad ID:", self.dyadID
        print "Payoff Condition:", self.payoff_structure
        print "STARTING ROUND...", self.round_n
        th = .2
        positions = [(-4, 2.5 + th), (4, 2.5 + th)]
        angles = [2 * np.pi, np.pi]
        #self.epucks = [IsmaEpuck(position=positions[i], angle=angles[i], frontIR=2, nother=2, nrewsensors=4) for i in range(n)]
        self.epucks = [IsmaEpuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=4) for i in range(n)]
        for e in self.epucks:
            if (self.payoff_structure == "high"):
                e.high_reward_value = 4
                # print "High reward value:", e.high_reward_value
            if (self.payoff_structure == "low"):
                e.high_reward_value = 2
                # print "High reward value:", e.high_reward_value
        #self.player1_pos = self.epucks[0].getPosition()
        #self.player2_pos = self.epucks[1].getPosition()
        #print "Player positions", self.player1_pos, self.player2_pos
        self.objs = []
        r = rand()
        if r < 0.5:
            self.high_reward_location = "top"
            addReward(self, pos=(0, 5 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)  # BIG reward
            addReward(self, pos=(0, 0 + th), vel=(0, 0), reward_type=1, bDynamic=False, bCollideNoOne=True)  # SMALL reward
        else:
            self.high_reward_location = "bottom"
            addReward(self, pos=(0, 0 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)  # BIG reward
            addReward(self, pos=(0, 5 + th), vel=(0, 0), reward_type=1, bDynamic=False, bCollideNoOne=True)  # SMALL reward
        # add some circles around the rewards
        addCircle(self, pos=(0, 5 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)
        addCircle(self, pos=(0, 0 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)
        # addWalls((0, 0), dx=3.75, dh=0.1, h=3, th=th)
        # addWalls((0, 0), dx=5.5, dh=0.8, h=3.4, th=th)

        """The contextual layer of each epuck selects the action that will take this round"""
        if (self.contextual is True):
            for e in self.epucks:
                e.contextual = True
                e.select_action()
                #self.epucks[0].action = self.epucks[0].contextual_layer.act(self.epucks[0].state)
                #self.epucks[1].action = self.epucks[1].contextual_layer.act(self.epucks[1].state)
            print "Player 1 selected action", self.epucks[0].action
            print "Player 2 selected action", self.epucks[1].action


    def update(self):
        """The contextual layer of each epuck selects the action that will take this round"""
        # self.epucks[0].action = self.epucks[0].contextual_layer.act(self.epucks[0].state)
        # self.epucks[1].action = self.epucks[1].contextual_layer.act(self.epucks[1].state)

        """Check epucks positions to see if they reached the rewards"""
        self.checkPositions()

        """Set out a timer to store data TWICE every second"""
        if self.timer == 0:
            self.timestep += 1
            # print "P1 absolute position:", self.p1_absposx, self.p1_absposy
            # print "P2 absolute position:", self.p2_absposx, self.p2_absposy
            # print "BITCH! timestep number: ", self.timestep
            #print "Avoid Epuck Weight:", self.epucks[0].avoid_epuck_w
            self.storedata()
        self.timer += 1
        self.timer = self.timer % 40  # ie: with 90, each 3 secs store data (in 30 FPS), with 60FPs: each 1,5 secs

        """If the round lasts more than 30 timesteps, the round ends"""
        if self.timestep > 60:
            print "TIMEOUT !! This round doesn't count!"
            self.timeout = True
            self.timeout_n += 1
            self.savedata()
            self.restart()

        """Update of epucks positions and gradient sensors: other and reward."""
        for e in self.epucks:
            e.update()
            pos = e.getPosition()

            for g in e.GradSensors:
                if(g.name == "other"):
                    centers = [o.getPosition() for o in self.epucks if o != e]
                    g.update(pos, e.getAngle(), centers)
                elif(g.name == "reward"):
                    centers = [o.position for o in self.objs[:1]]
                    g.update(pos, e.getAngle(), centers)
                    centers = [o.position for o in self.objs[-1:]]
                    g.update(pos, e.getAngle(), centers, extremes=1)

    def setMotors(self, epuck=0, motors=[10, 10]):
        self.epucks[epuck].motors = motors

    def constrain(self, n, minn, maxn):
        if n < minn:
            return minn
        elif n > maxn:
            return maxn
        else:
            return n

    def checkPositions(self):
        """Get the positions of both epucks and rewards"""
        self.player1_pos = self.epucks[0].getPosition()
        self.player2_pos = self.epucks[1].getPosition()

        """Calculate distance between both epucks"""
        #epuck_dist = dist(self.player1_pos, self.player2_pos)
        #epuck_dist = self.constrain(epuck_dist, 0, 2)
        # print "epuck distance:", epuck_dist

        #for e in self.epucks:
        #    e.avoid_epuck_w = 2.5 - epuck_dist

        # convert positions and angles
        #self.p1_absposx = round(self.player1_pos[0], 3)
        #self.p1_absposy = round(self.player1_pos[1], 3)
        self.p1_absposx = ((self.player1_pos[0] + 4)  * (540-180)/8) + 180
        self.p1_absposy = ((self.player1_pos[1] - 2.7) * (240-360)/2.5) + 240
        self.p1_absposx = round(self.p1_absposx, 3)
        self.p1_absposy = round(self.p1_absposy, 3)
        
        #self.p2_absposx = round(self.player2_pos[0], 3)
        #self.p2_absposy = round(self.player2_pos[1], 3)
        self.p2_absposx = ((self.player2_pos[0] - 4)  * (540-180)/8) + 540
        self.p2_absposy = ((self.player2_pos[1] - 2.7) * (240-360)/2.5) + 240
        self.p2_absposx = round(self.p2_absposx, 3)
        self.p2_absposy = round(self.p2_absposy, 3)

        self.player1_ang = int(np.rad2deg(self.epucks[0].getAngle())) + 90
        self.player2_ang = int(np.rad2deg(self.epucks[1].getAngle())) + 90
        self.player1_ang = self.player1_ang % 360
        self.player2_ang = self.player2_ang % 360
        # print "Player 1 angle: ", self.player1_ang
        # print "Player 2 angle: ", self.player2_ang

        hreward_pos = self.objs[0].position
        lreward_pos = self.objs[1].position
        # print "High and Low Reward pos", hreward_pos, lreward_pos
        # hreward_abs_posx = (hreward_pos[0] * (540-360)/4) + 360
        # hreward_abs_posy = (round(hreward_pos[1]-5.2, 3) * (120-360)/5) + 120   
        # lreward_abs_posx = (lreward_pos[0] * (540-360)/4) + 360
        # lreward_abs_posy = (round(lreward_pos[1]-0.2, 3) * (120-360)/5) + 360
        # print "High Reward positions", hreward_abs_posx, hreward_abs_posy
        # print "Low Reward positions", lreward_abs_posx, lreward_abs_posy

        """Calculate distance between both epucks and rewards"""
        p1hr_dist = dist(self.player1_pos, hreward_pos)
        p1lr_dist = dist(self.player1_pos, lreward_pos)
        p2hr_dist = dist(self.player2_pos, hreward_pos)
        p2lr_dist = dist(self.player2_pos, lreward_pos)

        if (p1hr_dist < config_data['reward_area']):  # REWARD AT 0.5, CIRCLE AT 1.25 
            if (p2hr_dist < 1.5):
                print "IT'S A TIE !! Both players get 0 points"
                self.ties_n += 1
                if (self.contextual is True): self.epucks[0].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[0].contextual_layer.update(2, 0, self.epucks[0].action)
                #self.epucks[0].state = 2
                if (self.contextual is True): self.epucks[1].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[1].contextual_layer.update(2, 0, self.epucks[1].action)
                #self.epucks[1].state = 2
                self.savedata()
                self.restart()
            else:
                print "Player 1 obtained the HIGH Reward !!"
                self.player1_wins += 1
                if (self.payoff_structure == "high"):
                    self.player1_score += 4
                    if (self.contextual is True): self.epucks[0].update_weights(1,4)   #STATE 1:High ; REWARD: 4
                    #self.epucks[0].contextual_layer.update(1, 4, self.epucks[0].action)
                    #self.epucks[0].state = 1
                if (self.payoff_structure == "low"):
                    self.player1_score += 2
                    if (self.contextual is True): self.epucks[0].update_weights(1,2)   #STATE 1:High ; REWARD: 2
                    #self.epucks[0].contextual_layer.update(1, 2, self.epucks[0].action)
                    #self.epucks[0].state = 1
                self.player2_score += 1
                if (self.contextual is True): self.epucks[1].update_weights(0,1)   #STATE 0:Low ; REWARD: 1
                #self.epucks[1].contextual_layer.update(0, 1, self.epucks[1].action)
                #self.epucks[1].state = 0
                self.savedata()
                self.restart()

        elif (p2hr_dist < config_data['reward_area']):
            if (p1hr_dist < 1.5):
                print "IT'S A TIE !! Both players get 0 points"
                self.ties_n += 1
                if (self.contextual is True): self.epucks[0].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[0].contextual_layer.update(2, 0, self.epucks[0].action)
                #self.epucks[0].state = 2
                if (self.contextual is True): self.epucks[1].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[1].contextual_layer.update(2, 0, self.epucks[1].action)
                #self.epucks[1].state = 2
                self.savedata()
                self.restart()
            else:
                print "Player 2 obtained the HIGH Reward !!"
                self.player2_wins += 1
                if (self.payoff_structure == "high"):
                    self.player2_score += 4
                    if (self.contextual is True): self.epucks[1].update_weights(1,4)   #STATE 1:High ; REWARD: 4
                    #self.epucks[1].contextual_layer.update(1, 4, self.epucks[1].action)
                    #self.epucks[1].state = 1
                if (self.payoff_structure == "low"):
                    self.player2_score += 2
                    if (self.contextual is True): self.epucks[1].update_weights(1,2)   #STATE 1:High ; REWARD: 2
                    #self.epucks[1].contextual_layer.update(1, 2, self.epucks[1].action)
                    #self.epucks[1].state = 1
                self.player1_score += 1
                if (self.contextual is True): self.epucks[0].update_weights(0,1)   #STATE 0:Low ; REWARD: 1
                #self.epucks[0].contextual_layer.update(0, 1, self.epucks[0].action)
                #self.epucks[0].state = 0
                self.savedata()
                self.restart()

        if (p1lr_dist < config_data['reward_area']):
            if (p2lr_dist < 1.5):
                print "IT'S A TIE !! Both players get 0 points"
                if (self.contextual is True): self.epucks[0].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[0].contextual_layer.update(2, 0, self.epucks[0].action)
                #self.epucks[0].state = 2
                if (self.contextual is True): self.epucks[1].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[1].contextual_layer.update(2, 0, self.epucks[1].action)
                #self.epucks[1].state = 2
                self.ties_n += 1
                self.savedata()
                self.restart()
            else:
                print "Player 1 obtained the LOW Reward !!"
                self.player2_wins += 1
                self.player1_score += 1
                if (self.contextual is True): self.epucks[0].update_weights(0,1)   #STATE 0:Low ; REWARD: 1
                #self.epucks[0].contextual_layer.update(0, 1, self.epucks[0].action)
                #self.epucks[0].state = 0
                if (self.payoff_structure == "high"):
                    self.player2_score += 4
                    if (self.contextual is True): self.epucks[1].update_weights(1,4)   #STATE 1:High ; REWARD: 4
                    #self.epucks[1].contextual_layer.update(1, 4, self.epucks[1].action)
                    #self.epucks[1].state = 1
                if (self.payoff_structure == "low"):
                    self.player2_score += 2
                    if (self.contextual is True): self.epucks[1].update_weights(1,2)   #STATE 1:High ; REWARD: 2
                    #self.epucks[1].contextual_layer.update(1, 2, self.epucks[1].action)
                    #self.epucks[1].state = 1
                self.savedata()
                self.restart()

        elif (p2lr_dist < config_data['reward_area']):
            if (p1lr_dist < 1.5):
                print "IT'S A TIE !! Both players get 0 points"
                if (self.contextual is True): self.epucks[0].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[0].contextual_layer.update(2, 0, self.epucks[0].action)
                #self.epucks[0].state = 2
                if (self.contextual is True): self.epucks[1].update_weights(2,0)   #STATE 2:Tie ; REWARD: 0
                #self.epucks[1].contextual_layer.update(2, 0, self.epucks[1].action)
                #self.epucks[1].state = 2
                self.ties_n += 1
                self.savedata()
                self.restart()
            else:
                print "Player 2 obtained the LOW Reward !!"
                self.player1_wins += 1
                self.player2_score += 1
                if (self.contextual is True): self.epucks[1].update_weights(0,1)   #STATE 0:Low ; REWARD: 1
                #self.epucks[1].contextual_layer.update(0, 1, self.epucks[1].action)
                #self.epucks[1].state = 0
                if (self.payoff_structure == "high"): 
                    self.player1_score += 4
                    if (self.contextual is True): self.epucks[0].update_weights(1,4)   #STATE 1:High ; REWARD: 4
                    #self.epucks[0].contextual_layer.update(1, 4, self.epucks[0].action)
                    #self.epucks[0].state = 1
                if (self.payoff_structure == "low"): 
                    self.player1_score += 2
                    if (self.contextual is True): self.epucks[0].update_weights(1,2)   #STATE 1:High ; REWARD: 2
                    #self.epucks[0].contextual_layer.update(1, 2, self.epucks[0].action)
                    #self.epucks[0].state = 1
                self.savedata()
                self.restart()

    def storedata(self):
        # if the round reaches the timeout condition, record it in the last value of the data 
        if (self.timeout is True):
            player1_data = [self.round_n, self.timestep, self.high_reward_location, self.player1, self.p1_absposx, self.p1_absposy, self.player1_ang, self.player1_score, self.timeout_n]
            player2_data = [self.round_n, self.timestep, self.high_reward_location, self.player2, self.p2_absposx, self.p2_absposy, self.player2_ang, self.player2_score, self.timeout_n]
        else:
            player1_data = [self.round_n, self.timestep, self.high_reward_location, self.player1, self.p1_absposx, self.p1_absposy, self.player1_ang, self.player1_score, 0]
            player2_data = [self.round_n, self.timestep, self.high_reward_location, self.player2, self.p2_absposx, self.p2_absposy, self.player2_ang, self.player2_score, 0]            

        self.round_data.append(player1_data)
        self.round_data.append(player2_data)

    def savedata(self):
        self.storedata()
        with open(self.dyadID+'.csv', 'wb') as myfile:
            writer = csv.writer(myfile)
            writer.writerows(self.round_data)

    def restart(self):
        if (self.timeout is False): self.round_n += 1
        self.timer = 0
        self.timestep = 0
        self.timeout = False

        """Restart the initial conditions and play again"""
        print "Total Score - Player 1:", self.player1_score, ", Player 2:", self.player2_score
        print "-------------------------------------------------"

        """Locate the epucks in their corresponding positions and angles"""
        th = .2
        positions = [(-4, 2.5 + th), (4, 2.5 + th)]
        angles = [2 * np.pi, np.pi]
        self.setMotors(epuck=0, motors=[0, 0])
        self.setMotors(epuck=1, motors=[0, 0])
        self.epucks[0].stop()
        self.epucks[1].stop()
        self.epucks[0].setPosition(positions[0])
        self.epucks[1].setPosition(positions[1])
        self.epucks[0].body.angle = angles[0]
        self.epucks[1].body.angle = angles[1]
        self.checkPositions()
        """Relocate the reward spots again"""
        r = rand()
        if r < 0.5:
            self.high_reward_location = "top"
            self.objs[0].position = (0, 5 + th)
            self.objs[1].position = (0, 0 + th)
        else:
            self.high_reward_location = "bottom"
            self.objs[0].position = (0, 0 + th)
            self.objs[1].position = (0, 5 + th)

        print "STARTING ROUND...", self.round_n
        """The contextual layer of each epuck selects the action that will take this round"""
        if (self.contextual is True):
            for e in self.epucks:
                e.select_action()
                #self.epucks[0].action = self.epucks[0].contextual_layer.act(self.epucks[0].state)
                #self.epucks[1].action = self.epucks[1].contextual_layer.act(self.epucks[1].state)
            print "Player 1 selected action", self.epucks[0].action
            print "Player 2 selected action", self.epucks[1].action

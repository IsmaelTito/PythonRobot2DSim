import numpy as np
import Box2D
import csv
from Box2DWorld import (world, createBox, createCircle, collisions)
from VectorFigUtils import vnorm, dist
from ExpRobotSetup import ExpSetupEpuck
from isma_epuck import IsmaEpuck
from numpy.random import rand


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


def addReward(who, pos=(0,0), vel=(0,0), reward_type=0, bDynamic=True, bCollideNoOne=False):
    if(reward_type == 0):
        name, r = "reward", 0.27
    else:
        name, r = "reward_small", 0.2

    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=10, name=name, r=r)
    obj.userData["energy"] = 1.0
    obj.userData["visible"] = 1.0
    obj.linearVelocity = vel
    who.objs.append(obj)


def addCircle(who, pos=(0,0), vel=(0,0), bDynamic=False, bCollideNoOne=False):
    r = 0.85  # PREVIOUSLY WAS 0.75
    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=0, r=r)
    obj.userData["energy"] = 1.0
    obj.userData["visible"] = 1.0
    obj.linearVelocity = vel
    # who.objs.append(obj)


class IsmaExpSetup(object):
    """Exp setup class with two epucks and two reward sites."""

    def __init__(self, n=1, payoff_type=0, debug=False):
        """Initialize the variables we need to store in each round of the game"""
        # HIGH: 4v1, LOW: 2v1
        if(payoff_type == 0):
            self.payoff_structure = "high"
        else:
            self.payoff_structure = "low"
        self.round_n = 1
        self.timestep = 0
        self.high_reward_location = "-"
        self.player1 = "host"
        self.player2 = "other"
        self.player1_pos = (0, 0)
        self.player2_pos = (0, 0)
        self.player1_ang = 0
        self.player2_ang = 0
        self.player1_score = 0
        self.player2_score = 0
        self.round_data = []
        """Create the two epucks, two rewards with circles around and walls."""
        global bDebug
        bDebug = debug
        print "-------------------------------------------------"
        print "Payoff Condition:", self.payoff_structure
        th = .2
        positions = [(-4, 2.5 + th), (4, 2.5 + th)]
        angles = [2 * np.pi, np.pi]
        self.epucks = [IsmaEpuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=4) for i in range(n)]
        # self.epucks = [Epuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=2) for i in range(n)]
        # print(self.epucks)
        self.objs = []
        r = rand()
        print r
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
        addWalls((0, 0), dx=5.5, dh=0.8, h=3.4, th=th)
        self.timer = 0

    def update(self):
        """Check epucks positions to see if they reached the rewards"""
        self.checkPositions()

        """Set out a timer to store data each 3 seconds"""
        if self.timer == 0:
            self.timestep += 1
            print "BITCH! timestep number: ", self.timestep
            self.storedata()
        self.timer += 1
        self.timer = self.timer % 90  # each 3 secs store data

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

    def checkPositions(self):
        """Get the positions of both epucks and rewards"""
        self.player1_pos = self.epucks[0].getPosition()
        self.player2_pos = self.epucks[1].getPosition()
        self.player1_ang = np.rad2deg(self.epucks[0].getAngle())
        self.player2_ang = np.rad2deg(self.epucks[1].getAngle())
        # print "Player 1 angle: ", self.player1_ang
        # print "Player 2 angle: ", self.player2_ang
        hreward_pos = self.objs[0].position
        lreward_pos = self.objs[1].position

        """Calculate distance between both epucks and rewards"""
        p1hr_dist = dist(self.player1_pos, hreward_pos)
        p1lr_dist = dist(self.player1_pos, lreward_pos)
        p2hr_dist = dist(self.player2_pos, hreward_pos)
        p2lr_dist = dist(self.player2_pos, lreward_pos)

        if (p1hr_dist < 0.5):  # REWARD AT 0.5, CIRCLE AT 1.25
            if (p2hr_dist < 1.25):
                print "IT'S A TIE !! Both players get 0 points"
                self.savedata()
                self.restart()
            else:
                print "Player 1 obtained the HIGH Reward !!"
                # print "Round Points - Player 1: 4 , Player 2: 1"
                if (self.payoff_structure == "high"): self.player1_score += 4
                if (self.payoff_structure == "low"): self.player1_score += 2
                self.player2_score += 1
                self.savedata()
                self.restart()

        if (p1lr_dist < 0.5):
            if (p2lr_dist < 1.25):
                print "IT'S A TIE !! Both players get 0 points"
                self.savedata()
                self.restart()
            else:
                print "Player 1 obtained the LOW Reward !!"
                self.player1_score += 1
                if (self.payoff_structure == "high"): self.player2_score += 4
                if (self.payoff_structure == "low"): self.player2_score += 2
                self.savedata()
                self.restart()

        if (p2hr_dist < 0.5):
            if (p1hr_dist < 1.25):
                print "IT'S A TIE !! Both players get 0 points"
                self.savedata()
                self.restart()
            else:
                print "Player 2 obtained the HIGH Reward !!"
                if (self.payoff_structure == "high"): self.player2_score += 4
                if (self.payoff_structure == "low"): self.player2_score += 2
                self.player1_score += 1
                self.savedata()
                self.restart()

        if (p2lr_dist < 0.5):
            if (p1lr_dist < 1.25):
                print "IT'S A TIE !! Both players get 0 points"
                self.savedata()
                self.restart()
            else:
                print "Player 2 obtained the LOW Reward !!"
                self.player2_score += 1
                if (self.payoff_structure == "high"): self.player1_score += 4
                if (self.payoff_structure == "low"): self.player1_score += 2
                self.savedata()
                self.restart()

    def storedata(self):
        # player1_data = [self.round_n, self.timestep, self.high_reward_location, self.player1, self.player1_pos[0], self.player1_pos[1], self.player1_ang, self.player1_score, 0]
        # player2_data = [self.round_n, self.timestep, self.high_reward_location, self.player2, self.player2_pos[0], self.player2_pos[1], self.player2_ang, self.player2_score, 0]
        player1_data = [self.round_n, self.timestep, self.high_reward_location, self.player1, self.player1_pos[0], self.player1_pos[1], self.player1_ang, self.player1_score, 0]
        player2_data = [self.round_n, self.timestep, self.high_reward_location, self.player2, self.player2_pos[0], self.player2_pos[1], self.player2_ang, self.player2_score, 0]

        self.round_data.append(player1_data)
        self.round_data.append(player2_data)

    def savedata(self):
        self.storedata()
        with open('test.csv', 'wb') as myfile:
            writer = csv.writer(myfile)
            writer.writerows(self.round_data)

    def restart(self):
        """Restart the initial conditions and play again"""
        print "-------------------------------------------------"
        self.round_n += 1
        self.timestep = 0

        print "STARTING ROUND...", self.round_n
        print "Total Score - Player 1:", self.player1_score, ", Player 2:", self.player2_score

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

        """Relocate the reward spots again"""
        r = rand()
        print r
        if r < 0.5:
            self.high_reward_location = "top"
            self.objs[0].position = (0, 5 + th)
            self.objs[1].position = (0, 0 + th)
        else:
            self.high_reward_location = "bottom"
            self.objs[0].position = (0, 0 + th)
            self.objs[1].position = (0, 5 + th)

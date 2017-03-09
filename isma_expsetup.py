import numpy as np
import Box2D
from Box2DWorld import (world, createBox, createCircle, collisions)
from ExpRobotSetup import ExpSetupEpuck
from isma_epuck import IsmaEpuck


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


class IsmaExpSetup(object):
    """Exp setup class with two epucks and two reward sites."""

    def __init__(self, n=1, debug=False):
        """Create the two epucks, two rewards and walls."""
        global bDebug
        bDebug = debug
        print "-------------------------------------------------"
        th = .2
        positions = [(-4, 2.5 + th), (4, 2.5 + th)]
        angles = [2 * np.pi, np.pi]
        self.epucks = [IsmaEpuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=4) for i in range(n)]
        # self.epucks = [Epuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=2) for i in range(n)]
        # print(self.epucks)
        # addWalls((0, 0), dx=3.75, dh=0.1, h=3, th=th)
        addWalls((0, 0), dx=5.5, dh=0.8, h=3.4, th=th)
        self.objs = []
        addReward(self, pos=(0, 5 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)  # BIG reward
        addReward(self, pos=(0, 0 + th), vel=(0, 0), reward_type=1, bDynamic=False, bCollideNoOne=True)  # SMALL reward

    def update(self):
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

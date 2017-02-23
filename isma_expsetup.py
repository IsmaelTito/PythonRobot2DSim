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
        createBox((x, y - 1 - dh + th), w=h + dh + wl + th, h=wl, bDynamic=False)
        createBox((x, y + 5 + dh + th), w=h + dh + wl + th, h=wl, bDynamic=False)
    if(bHoriz):
        createBox((x - dx - wl, y + yh - 1 + dh / 2 + th), w=wl, h=h + dh, bDynamic=False)
        createBox((x + dx + wl, y + yh - 1 + dh / 2 + th), w=wl, h=h + dh, bDynamic=False)


def addReward(who, pos=(0,0), vel=(0,0), bDynamic=True, bCollideNoOne=False):
    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=10, name="reward", r=0.2)
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
        positions = [(-3, 2 + th), (3, 2 + th)]
        angles = [2 * np.pi, np.pi]
        self.epucks = [IsmaEpuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=2) for i in range(n)]
        # self.epucks = [Epuck(position=positions[i], angle=angles[i], nother=2, nrewsensors=2) for i in range(n)]
        # print(self.epucks)
        addWalls((0, 0), dx=3.75, dh=0.1, h=3, th=th)
        self.objs = []
        addReward(self, pos=(0, 4 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)
        addReward(self, pos=(0, 0 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)

    def update(self):
        """Update of epucks positions and gradient sensors: other and reward."""
        for e in self.epucks:
            e.update()
            pos = e.getPosition()

            for g in e.GradSensors:
                if(g.name == "other"):
                    centers = [o.getPosition() for o in self.epucks if o != e]
                elif(g.name == "reward"):
                    centers = [o.position for o in self.objs]

                g.update(pos, e.getAngle(), centers)


    def setMotors(self, epuck=0, motors=[10, 10]):
        self.epucks[epuck].motors = motors
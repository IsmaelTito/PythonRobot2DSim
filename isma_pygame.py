import sys
sys.path.append('./_utils/')
import numpy as np
import pygame
import pygame.surfarray as surfarray
from pygame.locals import *


import PyGameUtils
import Box2DWorld
# from isma_epuck import IsmaEpuck
from isma_expsetup import IsmaExpSetup

box2dWH = (PyGameUtils.SCREEN_WIDTH, PyGameUtils.SCREEN_HEIGHT)

# ***************************
# PYGAME initialization
# ***************************
pygame.init()
PyGameUtils.setScreenSize(1000, 600)
box2dWH = (PyGameUtils.SCREEN_WIDTH, PyGameUtils.SCREEN_HEIGHT)

# flags = FULLSCREEN | DOUBLEBUF
# screen = pygame.display.set_mode(box2dWH, flags, 8)
screen = pygame.display.set_mode(box2dWH, 0, 32)
screen.set_alpha(None)
surfarray.use_arraytype('numpy')

pygame.display.set_caption('Epuck AWESOME Simulation')
clock = pygame.time.Clock()

exp = IsmaExpSetup(n=2, payoff_type=0, debug=True)
exp_time = 0
# epuck = IsmaEpuck()

running = True

while running:
    exp_time = pygame.time.get_ticks()/1000
    # left, right = epuck.prox_activations()
    # epuck.left_wheel = 1.0 - left   # to check
    # epuck.right_wheel = 1.0 - right   # to check

    # Check the event queue
    for event in pygame.event.get():
        if(event.type != pygame.KEYDOWN):
            continue

        if(event.key == pygame.K_LEFT):
            exp.setMotors(motors=[-10, 10])
        if(event.key == pygame.K_RIGHT):
            exp.setMotors(motors=[10, -10])
        if(event.key == pygame.K_UP):
            exp.setMotors(motors=[10, 10])
        if(event.key == pygame.K_DOWN):
            exp.setMotors(motors=[-10, -10])
        if(event.key == pygame.K_SPACE):
            exp.restart()

        if event.type == pygame.QUIT or event.key == pygame.K_ESCAPE:
            # The user closed the window or pressed escape
            running = False

    screen.fill((0, 0, 0, 0))

    # PyGameUtils.draw_contacts(screen,exp)
    PyGameUtils.draw_world(screen)

    exp.update()
    # exp.checkPositions()
    Box2DWorld.step()

    # PyGameUtils.draw_salient(screen, exp)

    pygame.display.flip()              # Flip the screen and try to keep at the target FPS
    clock.tick(30)
    pygame.display.set_caption("FPS: {:6.3}{}".format(clock.get_fps(), " "*5))

    # If one of the ePucks gets the High Reward, the other get instantly the Low Reward and the experiment ends
    # if exp.epucks[0].reward_score == 2:
        # exp.epucks[1].reward_score = 1
        # running = False

    # if exp.epucks[1].reward_score == 2:
        # exp.epucks[0].reward_score = 1
        # running = False

    # If the experiment reaches 10 seconds without any ePuck getting a reward, the experiment ends
    # if exp_time >= 10:
        # exp.epucks[0].reward_score = -1
        # exp.epucks[1].reward_score = -1
        # running = False


pygame.quit()
print('Done!')
# print "Epuck 1 FINAL reward score: ", exp.epucks[0].reward_score
# print "Epuck 2 FINAL reward score: ", exp.epucks[1].reward_score

from numpy.random import randint
from numpy import zeros, array
from numpy.random import randint

class GameExesTask(object):
    def __init__(self, high_rew=2.0, low_rew=1.0):
        self.actions = ["to_the_low", "to_the_high"]
        self.states = ["Low", "High", "Tie"]
        self.state_sensation_association = dict(Low=array([1., 0., 0.]), High=array([0., 1., 0.]), Tie=array([0., 0., 1]))
        
        self.initial_state = ["Tie", "Tie"]
        self.state = self.initial_state
        self.tie_rew = 0.0
        self.low_rew = low_rew
        self.high_rew = high_rew
        self.rewards = [self.tie_rew, self.tie_rew]
        
        
    def next_step(self, actions):
        action_ag1, action_ag2 = actions
        #action_ag1 = self.actions[randint(len(self.actions) - 1)] if action_ag1 == "none" else action_ag1
        #action_ag2 = self.actions[randint(len(self.actions) - 1)] if action_ag2 == "none" else action_ag2
    
        if action_ag1 == action_ag2:
            self.state = ["Tie", "Tie"]
            self.rewards = [self.tie_rew, self.tie_rew]
        elif action_ag1 == "to_the_high":
            self.state = ["High", "Low"]
            self.rewards = [self.high_rew, self.low_rew]
        elif action_ag1 == "to_the_low":
            self.state = ["Low", "High"]
            self.rewards = [self.low_rew, self.high_rew]
        else:
            print "Error in GameExesTask", action_ag1, action_ag2
            

        
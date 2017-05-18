from task import GameExesTask
from contextual_layer import ValueLearning, PolicyLearning, ContextualLayer

from copy import deepcopy
from numpy import array



class Logger(object):
    def __init__(self):
        self.logs = {}
    def add(self, topic, data):
        if topic not in self.logs:
            self.logs[topic] = [data]
        else:
            self.logs[topic].append(data)
    def get_log(self, topic):
        if topic not in self.logs:
            print "No data in " + topic
            return []
        else:
            return self.logs[topic]
    def clear(self):
        del self.logs
        self.logs = {}


class Experiment(object):
    def __init__(self, task, agents):
        self.task = task
        self.agents = agents
        self.logger = Logger()
    
    def run(self, n=100, debug=False):    
        for t in range(n):
            if debug:
                print t
            self.logger.add("td_error_hist", self.agents[0].value.td_error)
            self.logger.add("lr_act_hist", self.agents[0].policy.lr)
            self.logger.add("lr_val_hist", self.agents[0].value.lr)
            self.logger.add("value_hist", deepcopy([ag.value.values for ag in self.agents]))
            
            # Line below should call Contextual layer instead
            a = [ag.policy.decide(state) for ag, state in zip(self.agents, state_index(self.task))]
            ag_actions = [self.task.actions[aa] for aa in a]
            if debug:
                print "action_probs"
                print [ag.policy.probs for ag in self.agents]
                print ag_actions
            self.task.next_step(ag_actions)
            if debug:
                print self.task.state, self.task.rewards
            self.logger.add("rew_hist", self.task.rewards)
            [ag.update(state, reward, last_action) for ag, state, reward, last_action in zip(self.agents, state_index(self.task), self.task.rewards, a)]
            if debug:
                print "state values = ", [ag.value.values for ag in self.agents]



def make_experiment(payoff="low"):
    high_rew = 2.0 if payoff == "low" else 4.0
    task = GameExesTask(high_rew=high_rew)
    n_states = len(task.states)
    n_actions = len(task.actions)
    values = [ValueLearning(n_states) for _ in range (2)]
    policies = [PolicyLearning(n_states, n_actions) for _ in range (2)]
    agents = [ContextualLayer(v, p) for v,p in zip(values, policies)]
    experiment = Experiment(task, agents)   
    return experiment, task


def state_index(task):
    return [task.state_sensation_association[state].nonzero()[0][0] for state in task.state]




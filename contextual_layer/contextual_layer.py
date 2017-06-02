from numpy import ones, zeros, hstack, array
from copy import deepcopy
from numpy.random import randn
from scipy import stats


# Value learning
## lr: learning rate
## discount: discount factor

# PolicyLearning
## lr: learning rate
## alpha_laplace (maybe keep it as 1 and play on lr instead): alpha of the laplace succession rule   (ni + alpha) / (N + alpha * k), with N=sum(ni) and k = cardinality

class ValueLearning(object):
    def __init__(self, n_states, lr=0.1, discount=0.1):
        self.lr = lr
        self.lr_alpha = 0.3
        self.discount = discount
        self.values = 0. * ones(n_states)
        self.prev_state = 2  # Tie
        self.t = 0
        self.td_error = 0.

    def update(self, state, reward):
        # self.lr = 0.5 * (self.t + 1) ** (- self.lr_alpha)
        value_current_state = self.values[state]
        self.td_error = reward + self.discount * value_current_state - self.values[self.prev_state]
        self.values[self.prev_state] += self.lr * self.td_error
        self.prev_state = deepcopy(state)
        self.t += 1


class PolicyLearning(object):
    def __init__(self, n_states, n_actions, lr=0.3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha_laplace = 1
        self.lr = lr  # learning rate
        self.lr_alpha = 0.6
        self.action_count = array([[1. for _ in range(n_actions)] for _ in range(n_states)])
        self.t = 0

    def decide(self, state):
        self.probs = (self.action_count[state] + self.alpha_laplace) / (sum(self.action_count[state]) + self.alpha_laplace * self.n_actions)
        distr = stats.rv_discrete(values=(range(self.n_actions), self.probs))
        self.prev_state = deepcopy(state)
        return distr.rvs()

    def update(self, td_error, last_action):
        # self.lr = 0.5 * (self.t + 1) ** (- self.lr_alpha)
        self.action_count[self.prev_state, last_action] += self.lr * td_error
        self.action_count[self.prev_state, last_action] = max(self.action_count[self.prev_state, last_action], 0.0)
        self.t += 1


class ContextualLayer(object):
    def __init__(self, value, policy):
        self.value = value
        self.policy = policy
        
    def act(self, state):
        return self.policy.decide(state)
    
    def update(self, state, reward, last_action):
        self.value.update(state, reward)
        self.policy.update(self.value.td_error, last_action)
"""
Wrapper for ICU-Sepsis environment: https://github.com/icu-sepsis/icu-sepsis
MDP parameters (sepsis_dynamics.npz) from the developmental version (v0).
"""

import numpy as np

class SepsisEnv:
    def __init__(self, mdp_file):

        self.mdp_file = mdp_file

        data = np.load(mdp_file)

        self.p = data['tx_mat']
        self.R = data['r_mat']
        self.d_0 = data['d_0']

        self.action_space = self.p.shape[1]  # up and down
        # 0 - down
        # 1 - up
        self.state_space = np.arange(self.p.shape[0])

        self.n_dim = self.p.shape[0]
        self.n_actions = self.p.shape[1]

    def _make_one_hot(self, x):
        assert (type(x) == int)
        return np.eye(len(self.state_space))[x]

    def reset(self):
        self.state = np.random.choice(len(self.d_0), p=self.d_0)
        self.t = 0
        self.done = False

        return np.array([self.state])

    def step(self, action):
        next_state = np.random.choice(len(self.d_0), p=self.p[self.state, action])
        reward = self.R[self.state, action, next_state]
        if next_state in [len(self.d_0)-1, len(self.d_0)-2, len(self.d_0)-3]:
            self.done = True

        self.t += 1

        return np.array([next_state]), reward, self.done, {}
    

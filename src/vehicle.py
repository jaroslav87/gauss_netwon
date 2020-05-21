import numpy as np


class Vehicle:

    def __init__(self, x_in, dt, m, u, n):
        self.x = x_in
        self.m = m  # total mass
        self.u = u  # drag coefficient
        self.n = n  # size of input vector
        self.dt = dt  # sampling time
        self.v = np.zeros(n)  # velocity

    def simulate(self):
        self.v[0] = 0
        for i in range(0, self.n):
            if i == 0:
                self.v[i] = self.dt * self.x[i] / self.m
            else:
                self.v[i] = self.v[i-1] + (self.x[i]/self.m - (self.u/self.m)*self.v[i-1]*self.v[i-1])*self.dt
        print(self.v)
        return self.v


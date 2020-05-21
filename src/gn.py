import numpy as np

class GaussNetwon:

    def __init__(self, x_in, y_in):

        state_size = 2

        self.beta = [0.3, 0.9] # np.zeros(state_size)

        self.r_s = y_in.shape[0]  # count how many residuals we have
        self.x = x_in
        self.y = y_in
        self.J = np.zeros([self.r_s, state_size])
        self.r = np.zeros(self.r_s)
        self.Jr = np.zeros(state_size)

    def set_jacobian(self):
        for i in range(self.r_s):
            self.J[i, 0] = -self.x[i]/(self.beta[1] + self.x[i])
            self.J[i, 1] = self.x[i]*self.beta[0] / ((self.beta[1] + self.x[i])*(self.beta[1] + self.x[i]))
    def calc_res(self):
        err = 0
        for i in range(self.r_s):
            self.r[i] = self.y[i] - self.x[i]*self.beta[0]/(self.beta[1] + self.x[i])
            err = err + self.r[i]*self.r[i]
        return err


    def opt_step(self):
        self.set_jacobian()
        err = self.calc_res()
        Jt = np.transpose(self.J)
        self.beta = self.beta - np.matmul(np.matmul(np.linalg.inv(np.matmul(Jt, self.J)), Jt), self.r)
        print(self.beta)
        return err

    def opt(self, e):

        err = 100
        errp = 10
        while np.abs(err-errp) > e:
            errp = err
            err = self.opt_step()


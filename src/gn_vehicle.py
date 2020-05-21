import numpy as np

class GaussNetwonVehicle:

    def __init__(self, x_in, y_in, dt):

        state_size = 2  # how many parameters we search for

        self.beta = [10, 10]  # m and u, zeros here may lead to serious problems np.zeros(state_size)

        self.dt = dt  # sampling time in model
        self.r_s = y_in.shape[0]  # count how many residuals we have
        self.x = x_in  # opt input; time and force
        self.y = y_in   # target output; velocity
        self.J = np.zeros([self.r_s, state_size])  # Jacobian matrix
        self.r = np.zeros(self.r_s)  # residual vector
        #self.Jr = np.zeros(state_size)

    def set_jacobian(self):
        # here define the Jacobian matrix row - only the first, the rest will be calculated based on the first
        for i in range(0, self.r_s):
            if i == 0:
                self.J[i, 0] = (self.x[i]/(self.beta[0]*self.beta[0]))*self.dt - \
                           (self.beta[1]/(self.beta[0]*self.beta[0]))*self.dt*0
                self.J[i, 1] = (1.0 / self.beta[0]) * self.dt * 0
            else:
                self.J[i, 0] = (self.x[i] / (self.beta[0] * self.beta[0])) * self.dt - \
                               (self.beta[1] / (self.beta[0] * self.beta[0])) * self.dt * self.y[i - 1] * self.y[i - 1]
                self.J[i, 1] = (1.0/self.beta[0])*self.dt*self.y[i-1]*self.y[i-1]
    def calc_res(self):
        err = 0
        for i in range(0, self.r_s):
            if i == 0:
                self.r[i] = self.y[i] - self.dt*self.x[i]/self.beta[0] + (self.beta[1]/self.beta[0])*self.dt*0
            else:
                self.r[i] = self.y[i] - self.y[i - 1] - self.dt * self.x[i] / self.beta[0] + \
                            (self.beta[1] / self.beta[0]) * self.dt*self.y[i - 1] * self.y[i - 1]
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


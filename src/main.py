from src.gn import GaussNetwon
from src.vehicle import Vehicle
from src.gn_vehicle import GaussNetwonVehicle
import numpy as np
import matplotlib.pyplot as plt

xv = np.array([1000, 1200, 800, 500, 1800, 0, 1900, 600])
vehicle = Vehicle(xv, 20, 1500, 0.1, 8)

mu, sigma = 0, 0.5 # mean and standard deviation
s = np.random.normal(mu, sigma, 8)

yv = vehicle.simulate() + s

gnv = GaussNetwonVehicle(xv, yv, 20)
gnv.opt(0.0001)

vehicle = Vehicle(xv, 20, gnv.beta[0], gnv.beta[1], 8)
yvs = vehicle.simulate()

t = list(range(0, 160, 20))
plt.plot(t, yv, 'g', label='original')
plt.plot(t, yvs, 'r', label='simulated')
plt.ylabel('[m/s]')
plt.xlabel('[s]')
plt.title('vehicle speed')
plt.legend()
plt.show()

#x = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740], np.float)  # this might be a ndarray
#y = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317], np.float)

#gn = GaussNetwon(x, y)
#gn.opt(0.0001)



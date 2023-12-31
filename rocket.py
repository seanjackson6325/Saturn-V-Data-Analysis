# -*- coding: utf-8 -*-
"""
This is a program I made for a PHY224 lab at the University of Toronto.
Given some data about the Saturn V rocket, I developed methods to
accurately predict it's position.

I used a linear regression algorithm to find a linear function
desribing the rocket's position, and I also used the curve_fit
function provided by matplotlib. I compared the results of these
two functions on the plot, and the result was that there is little
to no difference between the two methods.
"""


import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def std_dev(data, mean):
    total = 0
    size = len(data)
    for i in range(size):
        total += (data[i] - mean)**2
    total /= size - 1
    return math.sqrt(total)

def std_error(std, size):
    return std / math.sqrt(size)


# Load and graph data from "rocket.csv"


rocket_data = np.loadtxt('rocket.csv', delimiter = ',')
N = len(rocket_data)

times = []
positions = []
uncertainties = []

for entry in rocket_data:
    times.append(entry[0])
    positions.append(entry[1])
    uncertainties.append(entry[2])
    
plt.scatter(x = times, y = positions)
plt.style.use("classic")
plt.title('Position vs. Time of  Saturn V rocket')
plt.xlabel("Time (hours)")
plt.ylabel("Position (kilometers)")
plt.errorbar(times, positions, yerr = uncertainties,
              color='red', ls = 'none')
plt.savefig("rocket_data_task1.jpg")


# Find the average speed from the data, as well as the
# standard error and standard deviations of the speed.


speeds_mean = sum(positions) / sum(times)

speeds = []

for i in range(1, N):
    d_position = abs(positions[i] - positions[i-1])
    d_time = abs(times[i]) - times[i-1]
    speeds.append(d_position / d_time)

speeds_std_dev = std_dev(speeds, speeds_mean)
speeds_std_error = std_error(speeds_std_dev, len(speeds))

print('\n-------- TASK 2 --------')
print('Std. Deviation of Speeds: ' + str(speeds_std_dev))
print('Std. Error of Speeds: ' + str(speeds_std_error))
print('------------------------\n')


# Find average position and time, use a linear regression model
# to calculate the speed and initial position of the rocket.
# this new function is called "estimate_pos"


positions_average = sum(positions) / len(positions)
times_average = sum(times) / len(times)

#calculating u

lr_numerator = 0
lr_denominator = 0
for i in range(N):
    dt = (times[i] - times_average)
    lr_numerator += dt * (positions[i] - positions_average)
    lr_denominator += dt**2

u = lr_numerator / lr_denominator
d_0 = positions_average - (u * times_average)

print('\n-------- TASK 3 --------')
print('u: ' + str(u))
print('d_0: ' + str(d_0))
print('------------------------\n')

def estimate_pos(t, d0, speed):
    return d0 + (speed * t)


estimated_positions = []
for i in range(N):
    estimated_positions.append(estimate_pos(times[i], d_0, u))


# Plot the linear regression model:


times_task_4 = np.array(times)
estimated_u_task_4 = estimate_pos(times_task_4, d_0, u)
plt.plot(np.array(times), estimated_u_task_4, color = 'green')


# Find chi (just a measure of how good of a fit the regression
# model is to the original data).


# X_2 Function:
def chi(measured, estimated, uncertainties):
    size = len(measured)
    total = 0
    for i in range(size):
        total += ((measured[i] - estimated[i])**2) / (uncertainties[i]**2)
    return total / (size - 2)

chi_fit_task_5 = chi(positions, estimated_positions, uncertainties)


print('\n-------- TASK 5 --------')
print('Chi (X_2) Output: ' + str(chi_fit_task_5))
print('------------------------\n')


# now, use curve_fit to fit the data instead of the linear regression.
# also, plot this new model:

def linear_function(x, m, b):
    return m*x + b

popt, pcov = curve_fit(\
    linear_function, times, positions, \
        p0 = [u, d_0], sigma = uncertainties, absolute_sigma = True)

pstd = np.sqrt(np.diag(pcov))

estimated_positions_task_6 = []
for i in range(N):
    estimated_positions_task_6.append(estimate_pos(times[i], popt[1], popt[0]))
chi_fit_task_6 = chi(positions, estimated_positions_task_6, uncertainties)

print('\n-------- TASK 6 --------')
print('New Parameter Estimates with curve_fit:')
print('\tStart Speed: ' + str(popt[0]) + ' km\hr')
print('\tStart Speed Uncertainty: +- ' + str(pstd[0]) + ' km\hr')
print('\tStart Position: ' + str(popt[1]) + ' km')
print('\tStart Position Uncertainty: +- ' + str(pstd[1]) + ' km\hr\n')
print('\tChi (X_2) Output: ' + str(chi_fit_task_6))
print('------------------------\n')

x = np.array(times)
y = linear_function(x, popt[0], popt[1])
plt.plot(x, y,  linestyle = 'dashed', color = 'Blue')
legend_strings = ['Position From Data', \
                  'Estimated Position (Linear Regression)',\
                  'Estimated Position (Curve Fit)']
plt.legend(legend_strings, loc = 'upper left')
plt.savefig("rocket_data_complete.jpg")

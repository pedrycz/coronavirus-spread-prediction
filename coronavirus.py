import sys
import matplotlib.pyplot as plt
from sir import sir_simulate
from data import load_data
sys.path.append("../astroABC/astroabc") # path to https://github.com/pedrycz/astroABC (forked from EliseJ)
from abc_class import *

data_file = 'china'
max_offset = 25
days_to_verify = 30
days_to_predict = 0

total_population, real_data_affected, real_data_infectious = load_data("data/" + data_file + ".txt")
real_data_size = len(real_data_affected)

# sampler helper function returning the ratio of infectious and recovered people (p - SIR ABC parameters)
def sir_sampler_helper(steps, p):
    Is, Rs = sir_simulate(steps, p[1], p[2], p[3])
    return np.multiply(Is, p[0]), np.multiply(Rs, p[0])

# sampler returning the ratio of affected and infectious people (steps_plus_offset - number of steps, p - SIR ABC parameters)
def coronavirus_simulate(p):
    Iratio, Rratio = sir_sampler_helper(steps_plus_offset, p)
    return np.add(Iratio, Rratio), Iratio

# sampler returning the ratio of affected people (global `steps_plus_offset` parameter required, p - SIR ABC parameters)
def coronavirus_simulate_affected(p):
    Iratio, Rratio = sir_sampler_helper(steps_plus_offset, p)
    return np.add(Iratio, Rratio)

# sampler returning the ratio of infectious people (global `steps_plus_offset` parameter required, p - SIR ABC parameters)
def coronavirus_simulate_infectious(p):
    Iratio, Rratio = sir_sampler_helper(steps_plus_offset, p)
    return Iratio

# cost functions
def squared_error_function(d, x):
    d_affected, d_infectious = d
    x_affected, x_infectious = x
    min = 1
    imin = 0
    for i in np.arange(0, max_offset, 1):
        current = 0
        if i > 0:
            if days_to_verify > 0:
                current = np.max(np.add(np.square(np.subtract(d_affected[i:-days_to_verify], x_affected[:-i])), np.square(np.subtract(d_infectious[i:-days_to_verify], x_infectious[:-i]))))
            else:
                current = np.max(np.add(np.square(np.subtract(d_affected[i:], x_affected[:-i])), np.square(np.subtract(d_infectious[i:], x_infectious[:-i]))))
        else:
            if days_to_verify > 0:
                current = np.max(np.add(np.square(np.subtract(d_affected[:-days_to_verify], x_affected)), np.square(np.subtract(d_infectious[:-days_to_verify], x_infectious))))
            else:
                current = np.max(np.add(np.square(np.subtract(d_affected, x_affected)), np.square(np.subtract(d_infectious, x_infectious))))
        if current < min:
            min = current
            imin = i
    return imin, min

def squared_error(d, x):
    imin, min = squared_error_function(d, x)
    return min

def compute_offset(d, x):
    imin, min = squared_error_function(d, x)
    return imin

# SIR ABC parameters :
# s% - total_affected_population / total_population
# i% - initially_affected_pupulation / total_affected_population
# b - illness rate
# k - healing rate
nparam = 4
param_names = ["s%", "i%", "b", "k"]
priorname = ["gamma", "uniform", "uniform", "uniform"]
hyperp = [[0.5, 100], [0, 0.1], [0.1, 1], [0, 0.1]]
prior = list(zip(priorname, hyperp))

# sampling parameters to adjust
npart = 20
niter = 10
tlevels = [1e-7, 1e-10]
threshold = 75
pert_kernel = 1
variance_method = 1
doublecheck_params = True
error_function = squared_error

# all merged properties passed to sampler
prop = {'tol_type': "exp", "verbose": 1, 'adapt_t': False, 'threshold': threshold,
        'pert_kernel': pert_kernel, 'variance_method': variance_method, 'dist_type': "user",
        'dfunc': error_function, 'outfile': None, 'doublecheck_params': doublecheck_params}

# create an instance of data sampler
sampler = ABC_class(nparam, npart, (real_data_affected, real_data_infectious), tlevels, niter, prior, **prop)

# invoke sampling
steps_plus_offset = real_data_size - days_to_verify - 1
sampler.sample(coronavirus_simulate)

# get sampling result
estimated_params = [np.mean(sampler.theta[niter-1][:,ii]) for ii in range(nparam)]

plot_offset = compute_offset((real_data_affected, real_data_infectious), coronavirus_simulate(estimated_params))

# simulate prediction
steps_plus_offset = real_data_size + days_to_predict
simulated_data_affected = coronavirus_simulate_affected(estimated_params)
simulated_data_infectious = coronavirus_simulate_infectious(estimated_params)

# generate plot data

plot_real_x = np.arange(0, real_data_size, 1)
plot_real_y_affected = np.multiply(real_data_affected, total_population)
plot_real_y_infectious = np.multiply(real_data_infectious, total_population)
plot_simulated_x = np.arange(plot_offset, steps_plus_offset + 1, 1)
plot_simulated_y_affected = np.multiply(simulated_data_affected[:-plot_offset], total_population)
plot_simulated_y_infectious = np.multiply(simulated_data_infectious[:-plot_offset], total_population)

# draw result
plt.ylabel("Population")
plt.xlabel("Number of days")
plt.plot(plot_real_x, plot_real_y_affected, "r", linewidth=1)
plt.plot(plot_real_x, plot_real_y_infectious, "r", linewidth=1, linestyle='--')
plt.plot(plot_simulated_x, plot_simulated_y_affected, "b")
plt.plot(plot_simulated_x, plot_simulated_y_infectious, "b", linestyle='--')
plt.axvline(x=real_data_size-days_to_verify-1, linestyle='--')
plt.legend(['real - total affected', 'real - affected', 'prediction - total affected', 'prediction - affected'], loc=4)
plt.show()

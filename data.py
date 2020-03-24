import numpy as np

def load_data(filepath):
    """
    load data from text file with the following structure:
    0th line: total population
    nth line: number of infected people in the nth day (cumulative) + number of infected people in the nth day (current)
    """

    total_population = np.loadtxt(filepath, max_rows=1, usecols=0)
    num_infected_cumulative, num_infected_current = np.loadtxt(filepath, skiprows=1, usecols=(0, 1), unpack=True)
    return total_population, np.divide(num_infected_cumulative, total_population), np.divide(num_infected_current, total_population)

import numpy as np

def load_data(filename):
    """
    load data from text file with the following structure:
    0th line: total population
    nth line: number of total affected people in the nth day, number of infectious people in the nth day (total affected - dead - recovered)
    """

    filepath = "data/" + filename + ".txt"
    total_population = np.loadtxt(filepath, max_rows=1, usecols=(0))
    num_affected, num_infectious = np.loadtxt(filepath, skiprows=1, usecols=(0, 1), unpack=True)
    return total_population, np.divide(num_affected, total_population), np.divide(num_infectious, total_population)

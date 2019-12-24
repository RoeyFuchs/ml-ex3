import numpy as np


def load_file(filename):
    return np.loadtxt(filename)



def shuffle_data(x_data, y_data):
    p = np.random.permutation(x_data.shape[0])
    x_data = x_data[p, :]
    y_data = y_data[p]
    return x_data, y_data


# split the data 80-20 - data set and validation set
def split(x_data, y_data):
    num = int(x_data.shape[0] * 0.2)
    x_valid = x_data[0:num, :]
    x_data = x_data[num:, :]
    y_valid = y_data[0:num]
    y_data = y_data[num:]
    return x_data, y_data, x_valid, y_valid


def normalize(array):
    norm = 255.0**2
    a = array/norm
    return a.copy()

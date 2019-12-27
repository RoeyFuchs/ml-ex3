from Utils import *
import sys
from scipy.special import softmax

relu_prime = lambda x: (x > 0) * 1


def relu(x):
    x[x < 0] = 0
    return x.copy()


def loss_func(x, y):
    epsilon = 1e-6
    return -np.log(x[int(y)] + epsilon)


def initialize(start_shape):
    num_of_nurons_1 = 128
    num_of_labels = 10
    uniform_range = 0.1
    # create weights and biases
    w1 = np.random.uniform(-uniform_range, uniform_range, ([start_shape, num_of_nurons_1]))
    w2 = np.random.uniform(-uniform_range, uniform_range, ([num_of_nurons_1, num_of_labels]))
    b1 = np.random.uniform(-uniform_range, uniform_range, ([1, num_of_nurons_1]))
    b2 = np.random.uniform(-uniform_range, uniform_range, ([1, num_of_labels]))

    return {'w1': w1.copy(), 'w2': w2.copy(), 'b1': b1.copy(), 'b2': b2.copy()}


def train(x_data, y_data, ephocs=50):
    x_data, y_data, x_valid, y_valid = split(x_data, y_data)  # split to training set and validation set
    init = initialize(x_data.shape[1])  # initialize weights and biases
    w1, b1, w2, b2 = [init[key] for key in ('w1', 'b1', 'w2', 'b2')]
    best_w1 = w1.copy()
    best_w2 = w2.copy()
    best_b1 = b1.copy()
    best_b2 = b2.copy()
    best = 0

    for _ in range(ephocs):
        x_data, y_data = shuffle_data(x_data, y_data)
        for x, y in zip(x_data, y_data):
            params = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
            fprop_vals = fprop(x, y, params)
            if (fprop_vals['loss'] > 0):
                bprop_vals = bprop(fprop_vals)
                #  update weights and biases
                learning_rate = 1e-3
                w1 -= learning_rate * bprop_vals['w1']
                w2 -= learning_rate * bprop_vals['w2']
                b1 -= learning_rate * bprop_vals['b1']
                b2 -= learning_rate * bprop_vals['b2']

        # run test on the validation set to store the best weights and biases
        correct = 0
        for x, y in zip(x_valid, y_valid):
            params = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
            fprop_vals = fprop(x, y, params)
            if np.argmax(fprop_vals['h2']) == y:
                correct += 1
        if (correct / y_valid.shape[0]) > best:
            best = correct / y_valid.shape[0]
            best_w1 = w1.copy()
            best_w2 = w2.copy()
            best_b1 = b1.copy()
            best_b2 = b2.copy()
    return {'w1': best_w1, 'w2': best_w2, 'b1': best_b1, 'b2': best_b2}


# Forward Propagation
def fprop(x, y, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    z1 = np.dot(w1.T, np.array([x]).T) + b1.T
    h1 = relu(z1.copy())
    z2 = np.dot(w2.T, h1) + b2.T
    h2 = softmax(z2.copy())
    loss = loss_func(h2.copy(), y.copy())
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


# back propagation - check what need to update
def bprop(fprop_vals):
    x, y, z1, h1, z2, h2, loss = [fprop_vals[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    yloss = np.zeros(h2.shape)
    yloss[int(y)] = 1
    dz2 = (h2 - yloss)
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2
    dz1 = np.dot(fprop_vals['w2'],
                 (h2 - yloss)) * relu_prime(z1)
    x = np.array([x])  # add dimension
    dW1 = np.dot(dz1, x)
    db1 = dz1
    return {'b1': db1.T, 'w1': dW1.T, 'b2': db2.T, 'w2': dW2.T, }


def predicte(x_test, best_values):
    w1, b1, w2, b2 = [best_values[key] for key in ('w1', 'b1', 'w2', 'b2')]
    for x in x_test:
        z1 = np.dot(w1.T, np.array([x]).T) + b1.T
        h1 = relu(z1.copy())
        z2 = np.dot(w2.T, h1) + b2.T
        h2 = softmax(z2.copy())
        pred = np.argmax(h2)
        with open('test_y', 'a') as file:
            file.write(str(pred) + '\n')


def main():
    x = normalize(load_file(sys.argv[1]))
    y = load_file(sys.argv[2])
    best_values = train(x, y)
    x_test = normalize(load_file(sys.argv[3]))
    predicte(x_test, best_values)


if __name__ == "__main__":
    main()

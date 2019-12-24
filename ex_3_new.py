from Utils import *
import sys
from scipy.special import softmax

# np.seterr(all='raise')

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
relu_prime = lambda x: (x > 0) * 1


def relu(x):
    x[x < 0] = 0
    return x.copy()


def loss_func(x, y):
    epsilon = 1e-10
    return -np.log(x[int(y)] + epsilon)


def initilize(start_shape):
    num_of_nurons_1 = 250
    num_of_nurons_2 = 100
    num_of_labels = 10
    uniform_range = 3
    w1 = np.random.uniform(-uniform_range, uniform_range, ([start_shape, num_of_nurons_1]))
    w2 = np.random.uniform(-uniform_range, uniform_range, ([num_of_nurons_1, num_of_nurons_2]))
    w3 = np.random.uniform(-uniform_range, uniform_range, ([num_of_nurons_2, num_of_labels]))

    # b1 = np.zeros([1, num_of_nurons_1])
    # b2 = np.zeros([1, num_of_nurons_2])
    # b3 = np.zeros([1, num_of_labels])

    b1 = np.random.uniform(-1, 1, ([1, num_of_nurons_1]))
    b2 = np.random.uniform(-1, 1, ([1, num_of_nurons_2]))
    b3 = np.random.uniform(-1, 1, ([1, num_of_labels]))

    return {'w1': w1.copy(), 'w2': w2.copy(), 'w3': w3.copy(), 'b1': b1.copy(), 'b2': b2.copy(), 'b3': b3.copy()}


def train(x_data, y_data, ephocs=40):
    x_data, y_data, x_valid, y_valid = split(x_data, y_data)

    init = initilize(x_data.shape[1])
    w1, b1, w2, b2, w3, b3 = [init[key] for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')]

    best_w1 = w1.copy()
    best_w2 = w2.copy()
    best_w3 = w3.copy()
    best_b1 = b1.copy()
    best_b2 = b2.copy()
    best_b3 = b3.copy()
    best = 0

    for _ in range(ephocs):
        print(_)
        x_data, y_data = shuffle_data(x_data, y_data)
        for x, y in zip(x_data, y_data):
            params = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3}
            fprop_vals = fprop(x, y, params)

            if (fprop_vals['loss'] > 0):
                bprop_vals = bprop(fprop_vals)

                #  update
                # learning_rate = 0.001
                learning_rate = 1 / 255
                w1 -= learning_rate * bprop_vals['w1']
                w2 -= learning_rate * bprop_vals['w2']
                w3 -= learning_rate * bprop_vals['w3']
                learning_rate = 0.001
                b1 -= learning_rate * bprop_vals['b1']
                b2 -= learning_rate * bprop_vals['b2']
                b3 -= learning_rate * bprop_vals['b3']

        correct = 0
        for x, y in zip(x_valid, y_valid):
            params = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3}
            fprop_vals = fprop(x, y, params)
            if np.argmax(fprop_vals['h3']) == y:
                correct += 1
        if (correct / y_valid.shape[0]) > best:
            best = correct / y_valid.shape[0]
            best_w1 = w1.copy()
            best_w2 = w2.copy()
            best_w3 = w3.copy()
            best_b1 = b1.copy()
            best_b2 = b2.copy()
            best_b3 = b3.copy()
        print("best: " + str(best))

    return {'w1': best_w1, 'w2': best_w2, 'w3': best_w3, 'b1': best_b1, 'b2': best_b2, 'b3': best_b3}


def fprop(x, y, params):
    w1, b1, w2, b2, w3, b3 = [params[key] for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')]
    z1 = np.dot(w1.T, np.array([x]).T) + b1.T
    h1 = relu(z1.copy())
    z2 = np.dot(w2.T, h1) + b2.T
    h2 = relu(z2.copy())
    z3 = np.dot(w3.T, h2) + b3.T
    h3 = softmax(z3.copy())
    loss = loss_func(h3.copy(), y.copy())
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3, 'h3': h3, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_vals):
    x, y, z1, h1, z2, h2, z3, h3, loss = [fprop_vals[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'z3',
                                                                      'h3', 'loss')]
    yloss = np.zeros(h3.shape)
    yloss[int(y)] = 1
    dz3 = (h3 - yloss)
    dW3 = np.dot(dz3, h2.T)

    db3 = dz3
    dz2 = np.dot(fprop_vals['w3'],
                 (h3 - yloss)) * relu_prime(z2)
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2

    temp = np.dot(fprop_vals['w2'], dz2)
    dz1 = temp[0] * relu_prime(z1.copy())
    x = np.array([x])
    dW1 = np.dot(dz1, x)
    db1 = dz1

    return {'b1': db1.T, 'w1': dW1.T, 'b2': db2.T, 'w2': dW2.T, 'w3': dW3.T, 'b3': db3.T}


def predicte(x_test, best_values):
    w1, b1, w2, b2, w3, b3 = [best_values[key] for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')]

    for x in x_test:
        z1 = np.dot(w1.T, np.array([x]).T) + b1.T
        h1 = relu(z1.copy())
        z2 = np.dot(w2.T, h1) + b2.T
        h2 = relu(z2.copy())
        z3 = np.dot(w3.T, h2) + b3.T
        h3 = softmax(z3.copy())
        pred = np.argmax(h3)
        with open('test_y', 'a') as file:
            file.write(str(pred)+'\n')






def main():
    x = normalize(load_file(sys.argv[1]))
    y = load_file(sys.argv[2])
    best_values = train(x, y)
    x_test = normalize(load_file(sys.argv[3]))
    predicte(x_test, best_values)


if __name__ == "__main__":
    main()

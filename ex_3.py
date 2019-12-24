from Utils import *
import sys

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
relu_prime = lambda x: (x > 0) * 1
j_nll = lambda v, y_correct: -np.log(v.T[int(y_correct)])


def soft_max_prime(S):
    return soft_max(S)*(1-soft_max(S))


def relu(x):
    x[x < 0] = 0
    return x.copy()


def soft_max(x):
    epsilon = 0.001
    y = np.sum(np.exp(x))
    if y == 0:
        y = epsilon
    return np.exp(x) / y


def main():
    x = normalize(load_file(sys.argv[1]))
    y = load_file(sys.argv[2])

    train(x, y)


def train(x_data, y_data, ephocs=10):
    x_data, y_data, x_valid, y_valit = split(x_data, y_data)

    funcs = {0: relu, 1: sigmoid, 2: sigmoid, 3: soft_max}
    div_funcs = {0: relu_prime, 1: sigmoid_prime, 2: sigmoid_prime, 3: soft_max_prime}

    number_of_labels = 10
    num_of_nurons_1 = 400
    num_of_nurons_2 = 200
    num_of_nurons_3 = 100

    w1 = np.random.rand(x_data.shape[1], num_of_nurons_1)/255
    w2 = np.random.rand(num_of_nurons_1, num_of_nurons_2)/255
    w3 = np.random.rand(num_of_nurons_2, num_of_nurons_3)/255
    w4 = np.random.rand(num_of_nurons_3, number_of_labels)/255

    b1 = 0
    b2 = 0
    b3 = 0
    for _ in range(ephocs):
        x_data, y_data = shuffle_data(x_data, y_data)
        for x, y in zip(x_data, y_data):
            x = np.array([x]).T
            z1 = np.dot(x.T, w1) + b1
            g1 = funcs[0](z1)
            z2 = np.dot(g1, w2) + b2
            g2 = funcs[1](z2)
            z3 = np.dot(g2, w3) + b3
            g3 = funcs[2](z3)
            z_final = np.dot(g3, w4)
            g_final = funcs[3](z_final)

            loss = j_nll(g_final, y)
            # print(loss)

            yt = np.zeros(g_final.shape)
            yt[0][int(y)] = 1

            dz4 = g_final - yt
            dw4 = np.dot(g3.T, dz4)

            temp = np.dot(dz4, w4.T)
            temp2 = div_funcs[2](z3)
            temp3 = temp * temp2
            dz3 = np.dot(dz4, w4.T) * div_funcs[2](z3)
            dw3 = np.dot(g2.T, dz3)

            dz2 = np.dot(dz3, w3.T) * div_funcs[1](z2)
            dw2 = np.dot(g1.T, dz2)

            dz1 = np.dot(dz2, w2.T) * div_funcs[0](z1)
            dw1 = np.dot(x, dz1)

            learning_rate = 1
            w1 += learning_rate * dw1.copy()
            w2 += learning_rate * dw2.copy()
            w3 += learning_rate * dw3.copy()
            w4 += learning_rate * dw4.copy()
            b1 = np.mean(dz1)
            b2 = np.mean(dz2)
            b3 = np.mean(dz3)
        correct = 0
        for x, y in zip(x_valid, y_valit):
            x = np.array([x]).T
            z1 = np.dot(x.T, w1) + b1
            g1 = funcs[0](z1)
            z2 = np.dot(g1, w2) + b2
            g2 = funcs[1](z2)
            z3 = np.dot(g2, w3) + b3
            g3 = funcs[2](z3)
            z_final = np.dot(g3, w4)
            g_final = funcs[3](z_final)
            if y == np.argmax(g_final):
                correct += 1
            else:
                print(str(np.argmax(g_final)) + "  " + str(y))
        print(correct / x_valid.shape[1])


'''
def train(x_data, y_data, epochs=15):
    W1 = np.random.rand(x_data.shape[1])
    b1 = np.random.rand(x_data.shape[1])
    W2 = np.random.rand(x_data.shape[1])
    b2 = np.random.rand(x_data.shape[1])
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    for _ in range(epochs):
        x_data, y_data = shuffle_data(x_data, y_data)
        for x, y in zip(x_data, y_data):
            fprop_cache = fprop(x, y, params)
            bprop_cache = bprop(fprop_cache)

            b1 = bprop_cache.get('b1')
            b2 = bprop_cache.get('b2')
            W1 = bprop_cache.get('W1')
            W2 = bprop_cache.get('W2')


def fprop(x, y, params):
    # np.seterr(all='raise')
    epsilon = 0.1
    # Follows procedure given in notes
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = sigmoid(z2)
    h2[h2==0] = epsilon
    h2[h2==1] -= epsilon
    h2 = np.abs(h2)

    loss = -(y * np.log(h2) + (1 - y) * np.log(1 - h2))
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    dz2 = (h2 - y)  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.dot(fprop_cache['W2'].T,
                 (h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

'''
if __name__ == "__main__":
    main()

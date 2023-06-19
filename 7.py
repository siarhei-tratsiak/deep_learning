import numpy as np


def neural_network(input, weights):
    hid = input.dot(weights[0])
    pred = hid.dot(weights[1])

    return pred


ih_wgt = np.array([[.1, .2, -.1],
                   [-.1, .1, .9],
                   [.1, .4, .1]]).T

hp_wgt = np.array([[.3, 1.1, -.3],
                   [.1, .2, .0],
                   [.0, 1.3, .1]]).T

weights = [ih_wgt, hp_wgt]

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])

pred = neural_network(input, weights)

print(pred)

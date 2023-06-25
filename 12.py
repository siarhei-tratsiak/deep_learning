def neural_network(input, weights):
    out = 0

    for i in range(len(input)):
        out += input[i] * weights[i]

    return out


def ele_mul(scalar, vector):
    out = [0, 0, 0]

    for i in range(len(out)):
        out[i] = vector[i] * scalar

    return out


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]

alpha = 0.01
weights = [0.1, 0.2, -0.1]
input = [toes[0], wlrec[0], nfans[0]]

for iter in range(3):
    pred = neural_network(input, weights)
    delta = pred - true
    error = delta ** 2
    weight_deltas = ele_mul(delta, input)

    print("Iteration: {}".format(iter + 1))
    print("Pred: {}".format(pred))
    print("Error: {}".format(error))
    print("Delta: {}".format(delta))
    print("Weights: {}".format(weights))
    print("Weight Deltas:")
    print(str(weight_deltas))
    print("\n")

    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]

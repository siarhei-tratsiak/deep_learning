def neural_network(input, weights):
    hid = vect_mat_mul(input, weights[0])
    pred = vect_mat_mul(hid, weights[1])

    return pred


def vect_mat_mul(vect, matrix):
    assert (len(vect) == len(matrix))

    output = [0, 0, 0]

    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])

    return output


def w_sum(a, b):
    assert (len(a) == len(b))

    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])

    return output


ih_wgt = [[.1, .2, -.1],
          [-.1, .1, .9],
          [.1, .4, .1]]

hp_wgt = [[.3, 1.1, -.3],
          [.1, .2, .0],
          [.0, 1.3, .1]]

weights = [ih_wgt, hp_wgt]

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]

pred = neural_network(input, weights)

print(pred)

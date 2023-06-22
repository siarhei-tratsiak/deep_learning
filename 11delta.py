weight, goal_pred, input = (0.0, 0.8, 0.5)

for iteration in range(100):
    pred = input * weight
    delta = pred - goal_pred
    error = delta ** 2
    weight_delta = delta * input
    weight -= weight_delta

    print('Error: {} Prdiction: {}'.format(error, pred))

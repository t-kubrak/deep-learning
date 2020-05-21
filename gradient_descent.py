input = 2
goal_pred = 0.8
weight = 0.5
alpha = 0.1

for iteration in range(20):
    pred = input * weight
    error = (pred - goal_pred) ** 2
    derivative = (pred - goal_pred) * input
    weight = weight - (derivative * alpha)

    print("Error: " + str(error) + " Prediction: " + str(pred))

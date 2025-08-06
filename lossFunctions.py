import numpy as np
from sklearn.metrics import log_loss


class LossFunction:
    def computeLoss(t, y):
        raise NotImplementedError("Activation function must be implemented in subclass")
    def derivative(t, y):
        raise NotImplementedError("Activation function must be implemented in subclass")


class MeanSquaredError(LossFunction):
    def computeLoss(t, y):
        return 0.5 * np.square(y-t)
    def derivative(t, y):
        return y-t

class CrossEntropy(LossFunction):
    def computeLoss(y_true, y_pred):
        return log_loss(y_true, y_pred)

# def computeLoss(y_pred, y_true):
#     squared_differences = np.square(y_true - y_pred)
#     mse = np.mean(squared_differences)
#     return mse
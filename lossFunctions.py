import numpy as np

class LossFunction:
    def computeLoss(self, t, y):
        raise NotImplementedError("Activation function must be implemented in subclass")
    def derivative(self, t, y):
        raise NotImplementedError("Activation function must be implemented in subclass")


class MeanSquaredError(LossFunction):
    def computeLoss(self, t, y):
        return 0.5 * np.square(y-t)
    def derivative(self, t, y):
        return y-t

class CrossEntropy(LossFunction):
    def computeLoss(self, t, y):
        # return -(np.)

#  Mean Squared Error (MSE):
# Formula: L = 1/2 * (y - t)^2, where 't' is the target value.
# Derivative: dL/dy = y - t. This is a simple subtraction.
# 2. Cross-Entropy Loss (for binary classification):
# Formula: L = -(t * log(y) + (1 - t) * log(1 - y)), where 't' is the target (0 or 1) and 'y' is the predicted probability.
# Derivative: dL/dy = (y - t) / (y * (1 - y))
# 3. Cross-Entropy Loss (for multi-class classification):
# Formula: L = - sum(t_i * log(y_i)), where 't' is a one-hot encoded vector of target labels, and 'y' is the output of the softmax function.
# Derivative: dL/dy_i = y_i - t_i for each output neuron 'i'.
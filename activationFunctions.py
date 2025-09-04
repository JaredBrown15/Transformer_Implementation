# Jared Brown
# 08/03/2025
import numpy as np
from tornado.gen import Return


class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError("Activation function must be implemented in subclass")

    def backward(self, dA, x):
        raise NotImplementedError("Activation function must be implemented in subclass") \

    def step(self, learning_rate, gradients):
        return

# x is dimension [batch_size x input_dim]
class ReLU(ActivationFunction):
    def forward(self, x, bias):
        return np.maximum(0, x)

    def backward(self, dA, x):
        # dA = dL = dX in the case of the output layer
        grad = np.where(x <= 0, 0, 1)
        return dA * grad        # returns dZ


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, dA, x):
        s = self.forward(x)  # or pass sigmoid output explicitly
        return dA * s * (1 - s)



class Softmax(ActivationFunction):
    @staticmethod
    def forward(x):
        # subtract max per column (batch)
        shift_x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    @staticmethod
    def backward(softmax_output, target_labels):
        return softmax_output - target_labels
        # return
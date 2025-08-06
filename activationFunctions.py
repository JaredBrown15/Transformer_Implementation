# Jared Brown
# 08/03/2025
import numpy as np
from tornado.gen import Return


class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError("Activation function must be implemented in subclass")

    def derivative(self, dA, x):
        raise NotImplementedError("Activation function must be implemented in subclass") \

    def step(self, learning_rate, gradients):
        return


class ReLU(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, dA, x):
        # dA = dL = dX in the case of the output layer
        return np.sign(dA)   # 0 if x=0, 1 if x>0
        # returns dZ


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, dA, x):
        return dA * (1 - dA)



class Softmax(ActivationFunction):
    def forward(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def derivative(self, softmax_output, target_labels):
        return softmax_output - target_labels
        # return
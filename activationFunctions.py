# Jared Brown
# 08/03/2025
import numpy as np
from tornado.gen import Return


class ActivationFunction:
    def forward(self, input):
        raise NotImplementedError("Activation function must be implemented in subclass")

    def derivative(self, input):
        raise NotImplementedError("Activation function must be implemented in subclass")



class ReLU(ActivationFunction):
    def forward(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.sign(input)   # 0 if input=0, 1 if input>0



class Sigmoid(ActivationFunction):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        return input * (1 - input)



class Softmax(ActivationFunction):
    def forward(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=0)

    def derivative(self, input):
        # TODO: Make Return
        # return
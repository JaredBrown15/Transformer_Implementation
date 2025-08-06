# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
# import activationFunctions as af
import activationFunctions as af
import copy

from activationFunctions import Softmax


class Transformer:
    def __init__(
            self,
            num_encoder_layers: int=6,
            num_decoder_layers: int=6,
            d_model: int=512,
            d_ff: int=2048,
            num_heads: int=8
    ):

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads




class TransformerEncoder:
    def __init__(
            self,
            num_layers: int=6,
            attention: bool=True,
            residual_connection: bool=True,
            d_model: int=512,
            num_heads: int = 8
    ):
        self.num_layers = num_layers
        self.attention = attention
        self.residual_connection = residual_connection
        self.d_model = d_model
        self.num_heads = num_heads


class TransformerDecoder:
    def __init__(
            self,
            num_layers: int = 6,
            attention: bool = True,
            residual_connection: bool = True,
            d_model: int = 512,
            num_heads: int=8
    ):
        self.num_layers = num_layers
        self.attention = attention
        self.residual_connection = residual_connection
        self.d_model = d_model
        self.num_head = num_heads


class FullyConnectedNetwork:
    def __init__(
            self,
            num_layers: int=1,
            d_hidden: int=32,
            d_in: int=4,
            d_out: int=3,
            dropout: float = 0.1,
            activation: str='ReLU',
            bias: bool=True
    ):
        try:
            activation_class = getattr(af, activation)
            self.activation = activation_class()
        except AttributeError:
            raise ValueError(f"No such activation function: {activation}")
        self.layer_neuron_vals = [None] * (num_layers + 1)
        self.layer_gradients = [None] * (num_layers + 1)
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.layers=[]
        for i in range(num_layers - 1):
            self.layers.append(copy.deepcopy(FullyConnectedLayer(d_in, d_hidden, self.dropout, bias)))
            self.layers.append(self.activation())
        self.layers.append(copy.deepcopy(FullyConnectedLayer(d_in, d_hidden, self.dropout, bias)))

    def forward(self, x, isTraining):
        for index, layer in enumerate(self.layers):
            if isTraining:
                self.layer_neuron_vals[index] = x
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for index, layer in enumerate(reversed(self.layers)):
            x = self.layer_neuron_vals[index]
            grad = layer.derivative(grad, x)
            self.layer_gradients[index] = grad
            grad = grad[0]  # This will be dX=dA from fully connected layer or dZ from activation layer
        return

    def step(self, learning_rate):
        for index, layer in enumerate(self.layers):
            layer = self.layers[index]
            gradients = self.layer_gradients[index]
            layer.step(learning_rate, gradients)

    def predict(self, x):
        output = self.forward(x, False)
        predictions = af.Softmax(x)
        return predictions

class FullyConnectedLayer:
    def __init__(self,
                 # activation,
                 input_dim: int=4,
                 # output_dim: int=128,
                 hidden_size: int=32,
                 dropout: float=0.1,
                 bias: bool=True
                 ):
        self.dropout = dropout
        # self.activation = activation
        self.w = np.random.randn(hidden_size, input_dim) * np.sqrt(2 / input_dim)
        if bias:
            self.b = np.full((hidden_size, 1), 0.01)

    def forward(self, x):
        return np.dot(self.w, x) + self.b


    def derivative(self, dZ, x):
        dW = np.dot(dZ, x.T)
        dB = dZ
        dX = np.dot(self.w.T, dZ)
        return [dX, dW, dB]

    def step(self, learning_rate, gradients):
        self.w += learning_rate * gradients
        self.b += learning_rate * gradients.sum(axis=0)
        return






















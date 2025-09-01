# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
# import activationFunctions as af
import activationFunctions as af
import copy
import math
from activationFunctions import Softmax
import torch.nn.functional as F


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
        self.num_heads = num_heads


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
        self.layer_neuron_vals = [None] * (num_layers*3 + 2)
        self.layer_gradients = [None] * (num_layers*3 + 1)
        self.num_layers = num_layers
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.bias = bias
        self.layers=[]
        self.layers.append(copy.deepcopy(LinearLayer(d_in, d_hidden, self.dropout, bias)))
        self.layers.append(self.activation)
        for i in range(num_layers-1):
            self.layers.append(copy.deepcopy(LinearLayer(d_hidden, d_hidden, self.dropout, bias)))
            self.layers.append(self.activation)
            if(dropout > 0):
                self.layers.append(DropoutLayer(self.dropout))
        self.layers.append(copy.deepcopy(LinearLayer(d_hidden, d_out, self.dropout, bias)))

    def forward(self, x, isTraining):
        for index, layer in enumerate(self.layers):
            if isTraining:
                self.layer_neuron_vals[index] = x
            x = layer.forward(x, isTraining)
        if isTraining:
            self.layer_neuron_vals[len(self.layer_neuron_vals)-1] = x
        return x

    def backward(self, grad):
        for index, layer in reversed(list(enumerate(self.layers))):
        # for index, layer in enumerate(reversed(self.layers)):
            x = self.layer_neuron_vals[index]
            grad = layer.derivative(grad, x)
            self.layer_gradients[index] = grad
            if len(grad) == 3:
                grad = grad[0]  # This will be dX=dA from fully connected layer or dZ from activation layer
        return

    def step(self, learning_rate):
        for index, layer in enumerate(self.layers):
            layer = self.layers[index]
            gradients = self.layer_gradients[index]
            layer.step(learning_rate, gradients)

    def predict(self, x):
        output = self.forward(x, False)
        predictions = af.Softmax.forward(output)
        return predictions

class LinearLayer:
    """Fully connected linear layer

    Args:
        input_dim (int): Dimension of input to layer ()
        hidden_size (int): The width of the rectangle.

    Returns:
        float: The area of the rectangle.

    Raises:
        ValueError: If `length` or `width` is negative.
    """
    def __init__(self,
                 input_dim: int=4,
                 hidden_size: int=32,
                 dropout: float=0.1,
                 bias: bool=True
                 ):
        self.dropout = dropout
        self.bias = bias
        self.w = np.random.randn(hidden_size, input_dim) * np.sqrt(2 / input_dim)   #Kaiming/He initialization
        if self.bias:
            self.b = np.full((hidden_size, 1), 0.01)
        else:
            self.b = np.full((hidden_size, 1), 0)

    def forward(self, x, is_training):
            # x is dimensions [input_dim  x batch_size]
            # w is dimensions [hidden_size x input_dim]
            return np.dot(self.w, x) + self.b

    def derivative(self, dZ, x):
        batch_size = x.shape[0]             # Assumes x is size [input_dim x batch_size]
        dW = np.dot(dZ, x.T) / batch_size   # dZ dimensions [hidden_size x input_dim]
        dB = np.sum(dZ, axis=1, keepdims=True) / batch_size
        dX = np.dot(self.w.T, dZ)
        return [dX, dW, dB]

    def step(self, learning_rate, gradients):
        self.w -= learning_rate * gradients[1]
        if self.bias:
            self.b -= learning_rate * gradients[2]
        return


class DropoutLayer:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, input, training):
        if training:
            self.mask = (np.random.rand(*input.shape) > self.rate).astype(float)
            return (input * self.mask) / (1 - self.rate)
        else:
            return input  # no dropout in inference

    def derivative(self, dZ, input):
        return (dZ * self.mask) / (1 - self.rate)

    def step(self, learning_rate, gradients):
        return


class MultiHeadAttentionLayer:
    def __init__(
            self,
            d_model,
            num_heads,
            dropout=0.0,
            bias = True):
        assert d_model % num_heads == 0, "model dimension must be divisible by number of heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.d_head = self.d_model // self.num_heads

        self.w_q = LinearLayer(d_model, d_model, self.dropout, bias)
        self.w_k = LinearLayer(d_model, d_model, self.dropout, bias)
        self.w_v = LinearLayer(d_model, d_model, self.dropout, bias)
        self.w_output = LinearLayer(d_model, d_model, self.dropout, bias)
        self.scale_value = float(1.0 / math.sqrt(self.dk))

    def forward(self, x, is_training):
        # x is dimensions [batch_size x sequence_len x d_model]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Linear Projections - dimensions [batch_size x sequence_len x d_model]
        q = self.w_q.forward(x, True)
        k = self.w_k.forward(x, True)
        v = self.w_v.forward(x, True)

        # Reshape projections to add a dimension, allowing for multiple heads
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_model)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_model)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_model)

        # Transpose to get correct order of dimensions (necessary for matmul operation when calculating attention scores)
        # [batch_size, num_heads, seq_len, d_head]
        q = q.transpose(1, 2)       # This swaps dimensions 1 and 2
        k = k.transpose(1, 2) 
        v = v.transpose(0, 2, 1, 3) # Another way of doing it...

        # Results in [batch_size, num_heads, seq_len, d_head]
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale_value
        attention_weights = F.softmax(attention_scores, axis=-1)    # Must apply just to final dimension... Using pytorch softmax here b/c easier

        context = context.transpose(0, 2, 1, 3)  # [batch, seq_len, heads, d_head]
        context = context.reshape(batch_size, seq_len, self.d_model)

        













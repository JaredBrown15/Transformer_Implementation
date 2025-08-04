# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
# import activationFunctions as af
import activationFunctions as af
import copy

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


class FeedForwardNetwork:
    def __init__(
            self,
            num_layers: int=1,
            d_ff: int=128,
            d_in: int=128,
            d_out: int=128,
            dropout: float = 0.1,
            activation: str='ReLU',
            bias: bool=True
    ):
        try:
            activation_class = getattr(af, activation)
            self.activation = activation_class()
        except AttributeError:
            raise ValueError(f"No such activation function: {activation}")
        self.num_layers = num_layers,
        self.d_ff = d_ff,
        self.dropout = dropout
        self.layers = [copy.deepcopy(FeedForwardLayer(d_in, d_out, d_ff, self.dropout, bias)) for i in range(num_layers)]
        self.layers.append(af.ReLU())

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, loss_error):
        

class FeedForwardLayer:
    def __init__(self,
                 # activation,
                 input_dim: int=128,
                 output_dim: int=128,
                 hidden_size: int=128,
                 dropout: float=0.1,
                 bias: bool=True
                 ):
        self.hidden_size = hidden_size,
        self.dropout = dropout,
        # self.activation = activation
        self.w = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        if bias:
            self.b = np.full(output_dim, 0.01)

    def forward(self, input):
        return np.matmul(input, self.w) + self.b


    def derivative(self, dl_dy, x):
        dl_dW = np.dot(dl_dy, x.T)
        dl_db = dl_dy
        dl_dx = np.dot(dl_dy, self.w.T)
        return np[dl_dw, dl_dw, dl_dx]

























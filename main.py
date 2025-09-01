from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler

import transformer
import activationFunctions
import lossFunctions
import numpy as np
import pandas as pd

from activationFunctions import ActivationFunction

irisData = pd.read_csv('iris.csv')
irisData.drop(columns=['Id'], inplace=True)
target_column = irisData.columns[-1]
y = pd.get_dummies(irisData[target_column]).to_numpy()
X = irisData.drop(columns=[target_column]).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.T)
X_test  = scaler.transform(X_test.T)


losses = []
learning_rate = 0.0005


def train(x, y, network):
    # for entry in x:
    output = network.forward(x, True)
    predictions = activationFunctions.Softmax.forward(output)
    losses.append(lossFunctions.CrossEntropy.computeLoss(y.T, predictions.T))
    dA = activationFunctions.Softmax.derivative(predictions, y)
    network.backward(dA)
    network.step(learning_rate)


def main():

    # denseNetwork = transformer.FullyConnectedNetwork()
    denseNetwork = transformer.FullyConnectedNetwork(
        num_layers = 4,
        d_hidden = 64,
        d_in = 4,
        d_out = 3,
        dropout = 0.15,
        activation = 'ReLU',
        bias = True)

    iters = 5000
    for i in range(iters):
        # NOTE: X should be [batch_size x input_dim (num_features)]
        train(X_train.T, y_train, denseNetwork)

    predictions_test = denseNetwork.predict(X_test.T)
    # predictions_test = denseNetwork.predict(output_test)
    loss = lossFunctions.CrossEntropy.computeLoss(y_test.T, predictions_test.T)

    num_classes, num_samples = predictions_test.shape
    # Initialize one-hot matrix with zeros
    one_hot = np.zeros_like(predictions_test, dtype=int)
    # Find the index of max probability per sample
    max_indices = np.argmax(predictions_test, axis=0)  # shape: (num_samples,)
    # Set the max index position to 1 for each sample
    one_hot[max_indices, np.arange(num_samples)] = 1




    precision, recall, f1, support = precision_recall_fscore_support(y_test.T,one_hot.T)
    print(precision, recall, f1, support)
    accuracy = accuracy_score(y_test.T, one_hot.T)
    print("Accuracy: " + str(accuracy))
    indices = np.arange(len(losses))
    plt.plot(indices, losses)
    plt.show()


main()
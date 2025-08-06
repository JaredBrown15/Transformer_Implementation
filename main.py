from sklearn.model_selection import train_test_split

import transformer
import activationFunctions
import lossFunctions

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

losses = []
learning_rate = 0.01


def train(x, y, network):
    # for entry in x:
    output = network.forward(x, True)
    predictions = activationFunctions.Softmax.forward(output)
    losses.append(lossFunctions.CrossEntropy.computeLoss(y, predictions))
    dA = activationFunctions.Softmax.derivative(predictions, y)
    network.backward(dA)
    network.step(learning_rate)


def main():

    denseNetwork = transformer.FullyConnectedNetwork()
    iters = 10
    for i in range(iters):
        train(X_train, y_train, denseNetwork)

    print(losses)
    output_test = denseNetwork.predict(X_test)
    predictions_test = denseNetwork.predict(output_test)
    loss = lossFunctions.CrossEntropy.computeLoss(y_test, predictions_test)
    print(loss)


main()
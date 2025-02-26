import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    num_samples = train_x.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        batch_x = train_x[batch_indices]
        batch_y = train_y[batch_indices]

        # Ensure `batch_y` keeps its shape (batch_size, 1) for regression tasks
        if len(batch_y.shape) == 1:  # If it's 1D
            batch_y = batch_y.reshape(-1, 1)

        yield batch_x, batch_y


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        tanh_x = self.forward(x)
        return 1 - (tanh_x**2)
    

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        softmax_x = self.forward(x)
        return softmax_x * (1 - softmax_x)

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * ((y_pred - y_true) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
    

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-12, 1.0)  # Prevent log(0) errors
        return -np.sum(y_true * np.log(y_pred), axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-12, 1.0)  # Prevent log(0) errors
        return y_pred - y_true


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biaes
        self.W = self.glorot_uniform(fan_in, fan_out)
        self.b = np.zeros((1, fan_out))


    def glorot_uniform(self, fan_in, fan_out):
        """
        Glorot Uniform Initialization.
        
        :param fan_in: Number of input units.
        :param fan_out: Number of output units.
        :return: Initialized weight matrix.
        """
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """        
        z = h @ self.W + self.b
        self.activations = self.activation_function.forward(z)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        z = h @ self.W + self.b
        dO_dZ = self.activation_function.derivative(z)

        dL_dW = h.T @ (delta * dO_dZ)
        dL_db = dL_db = np.sum(delta * dO_dZ, axis=0, keepdims=True)
        self.delta = (delta * dO_dZ) @ self.W.T
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []

        # Init delta as the loss gradient, which is dL_dY_hat
        delta = loss_grad

        # Iterate backwards through the layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            # Get input to the current layer
            if i == 0:
                h = input_data
            else:
                h = self.layers[i-1].activations

            # Call backward prop for the layer
            dL_dW, dL_db = layer.backward(h, delta)

            # Store gradients
            dl_dw_all.insert(0, dL_dW)
            dl_db_all.insert(0, dL_db)

            # Update delta for next layer
            delta = layer.delta

        return dl_dw_all, dl_db_all


    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                # Feedforward
                y_pred = self.forward(batch_x)

                # Compute loss and gradient
                loss = loss_func.loss(batch_y, y_pred)
                loss_grad = loss_func.derivative(batch_y, y_pred)
                epoch_loss += np.sum(loss)

                # Backpropagation
                dl_dw_all, dl_db_all = self.backward(loss_grad, batch_x)

                # Weight and bias update
                for layer, dW, db in zip(self.layers, dl_dw_all, dl_db_all):
                    layer.W -= dW * learning_rate
                    layer.b -= db * learning_rate

            # Average training loss for epoch
            avg_train_loss = epoch_loss / train_x.shape[0]
            training_losses.append(avg_train_loss)

            # Validation loss
            val_pred = self.forward(val_x)
            val_loss = np.mean(loss_func.loss(val_y, val_pred))
            validation_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f} - Validation Loss: {val_loss:.4f}")
                
        return training_losses, validation_losses
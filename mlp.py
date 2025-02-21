import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    pass


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

        # Compute diagonal matrix diag(S)
        diag_matrix = np.einsum('bi,ij->bij', softmax_x, np.eye(softmax_x.shape[1]))

        # Compute outer product S @ S^T
        outer_product = np.einsum('bi,bj->bij', softmax_x, softmax_x)

        # Compute Jacobian as diag(S) - S @ S^T
        jacobian = diag_matrix - outer_product

        return jacobian

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
        return 0.5 * (y_pred - y_true) ** 2

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true
    

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-12, 1.0)  # Prevent division by zero
        return - (y_true / y_pred) / len(y_true)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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
        self.W = self.glorot_uniform(fan_in, fan_out)  # weights
        self.b = np.zeros((1, fan_out))  # biases

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



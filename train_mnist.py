# %%
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from mlp import MultilayerPerceptron, Layer, Sigmoid, Tanh, Relu, Softmax, Linear, SquaredError, CrossEntropy
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and flatten input data
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Split into training (80%) and validation (20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define MLP architecture
mlp = MultilayerPerceptron([
    Layer(784, 128, Relu()),
    Layer(128, 64, Relu()),
    Layer(64, 32, Relu()),
    Layer(32, 16, Relu()),
    Layer(16, 10, Softmax())
])

# Define loss function
loss_func = CrossEntropy()

# Train the MLP model
training_losses, validation_losses = mlp.train(
    train_x=x_train, 
    train_y=y_train,
    val_x=x_val, 
    val_y=y_val,
    loss_func=CrossEntropy(),
    learning_rate=0.01, 
    batch_size=32, 
    epochs=20
)

# Evaluate on test set
y_test_pred = mlp.forward(x_test)
test_loss = np.mean(loss_func.loss(y_test, y_test_pred))
accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))

print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label="Training Loss", linewidth=2)
plt.plot(validation_losses, label="Validation Loss", linewidth=2)

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)

plt.show()
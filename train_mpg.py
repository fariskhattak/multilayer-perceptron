from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import MultilayerPerceptron, Layer, SquaredError, Relu, Linear, Sigmoid, Softmax
import matplotlib.pyplot as plt
import numpy as np

# Fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# Data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# One-hot encode categorical variables
data = pd.get_dummies(X, columns=["cylinders", "model_year", "origin"], dtype=int)

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

# Ensure correct shape
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1,1)

# Define network architecture
mlp = MultilayerPerceptron(layers=[
    Layer(X_train.shape[1], 64, Relu()),
    Layer(64, 32, Relu()),
    Layer(32, 1, Linear())
])

# Define loss function
loss_func = SquaredError()

# Train the MLP
training_losses, validation_losses = mlp.train(
    train_x=X_train,
    train_y=y_train,
    val_x=X_val,
    val_y=y_val,
    loss_func=loss_func,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)

# Evaluate on test set
y_test_pred = mlp.forward(X_test)
test_loss = np.mean(loss_func.loss(y_test, y_test_pred))

print("\nFinal Test Loss (MSE):", test_loss)

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label="Training Loss", linewidth=2)
plt.plot(validation_losses, label="Validation Loss", linewidth=2)

# Labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)

plt.show()
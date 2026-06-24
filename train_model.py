import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load MNIST Dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images to 784 features
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("Training Samples :", x_train.shape)
print("Testing Samples  :", x_test.shape)

# -----------------------------
# Build Neural Network
# -----------------------------
model = Sequential([
    Input(shape=(784,)),

    Dense(256, activation="relu"),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(10, activation="softmax")
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy :", accuracy * 100)

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)

model.save("model/mnist_model.keras")

print("\nModel Saved Successfully!")

# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Loss Graph
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
import json
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

use_saved_model = False  # Set to True to load a previous model if available


def parse_sample(sample):
    """
    Parses a single sample from the JSON file.
    Expects each sample to have:
      - "x": a pickled numpy array representing an 8x8 EEG data matrix.
      - "y": the label.
    Returns the decoded numpy array.
    """
    arr = pickle.loads(sample["x"].encode('latin1'))
    if arr.shape != (8, 8):
        raise ValueError(f"Sample shape is {arr.shape} but expected (8,8)")
    return arr

def load_and_prepare_data(file_path):
    """
    Loads the JSON file and parses each sample for a 1D CNN.
    Each sample must have keys "x": the pickled numpy array; and "y": the label.
    Returns training and test data, the number of classes, and the original unique labels.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)
    
    X = []
    y = []
    for key, sample in data_dict.items():
        try:
            parsed = parse_sample(sample)
            X.append(parsed)
            y.append(sample["y"])
        except Exception as e:
            print(f"Skipping sample {key} due to error: {e}")
    
    X = np.array(X).astype("float32") 
    y = np.array(y)
    
    # Replace NaN or Inf values.
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize each sample
    X_min = X.min(axis=(1,2), keepdims=True)
    X_max = X.max(axis=(1,2), keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    
    print("Raw labels before mapping:", np.unique(y))
    
    # Map labels to a contiguous range.
    unique_labels, y_mapped = np.unique(y, return_inverse=True)
    num_classes = len(unique_labels)
    print("Unique labels after mapping (in order):", unique_labels)
    
    # One-hot encode labels.
    y_encoded = to_categorical(y_mapped, num_classes)
    
    # Split into training and test sets (80% train, 20% test).
    X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, Y_train, Y_test, num_classes, unique_labels

# Set the file path to your JSON dataset.
file_path = "data/output-balanced-classes-10000-or-more.json"
X_train, X_test, Y_train, Y_test, num_classes, unique_labels = load_and_prepare_data(file_path)

print("Training data shape:", X_train.shape) 
print("Test data shape:", X_test.shape)
print("Original unique labels:", unique_labels)
print("Number of classes:", num_classes)

# For a 1D CNN, we treat each 8x8 sample as 8 time steps with 8 features
# The input shape for Conv1D is (time_steps, features) = (8, 8).

# Model configuration:
model_save_path = os.path.join("models", "OutModel.h5")

if use_saved_model and os.path.exists(model_save_path):
    print(f"Loading existing model from: {model_save_path}")
    model = load_model(model_save_path)
    history = None
else:
    print("Creating a new model...")
    model = Sequential([
        Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(8, 8), padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),
        BatchNormalization(),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # Output layer with {num_classes} units.
    ])
    optimizer = Adam(learning_rate=0.0003, clipvalue=1.0)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))
    
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved as {model_save_path}")

model.summary()

# Evaluate the model.
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)

# Compute confusion matrix.
y_true = np.argmax(Y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Plot cost (loss) over epochs if training history is available.
if history is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Cost over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

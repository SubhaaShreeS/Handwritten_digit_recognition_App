import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from PIL import Image, ImageEnhance, ImageOps
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Load and preprocess MNIST dataset
(x_mnist, y_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_mnist = x_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_mnist = to_categorical(y_mnist, 10)
x_test_mnist = x_test_mnist.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test_mnist = to_categorical(y_test_mnist, 10)

print("MNIST loaded:", x_mnist.shape)

# 2. Load and preprocess custom handwritten digits
def preprocess_custom_image(img_path):
    img = Image.open(img_path).convert('L')
    img = ImageOps.invert(img) if np.mean(img) < 127 else img
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    img = img.resize((28, 28), Image.LANCZOS)
    return np.array(img).astype('float32') / 255.0

custom_dir = "/home/fleettrack/subhaashree/digit_reco_multi_backend/custom_dataset"
x_custom = []
y_custom = []

for digit in range(10):
    digit_dir = os.path.join(custom_dir, str(digit))
    if os.path.exists(digit_dir):
        for img_file in os.listdir(digit_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(digit_dir, img_file)
                try:
                    img_array = preprocess_custom_image(img_path)
                    x_custom.append(img_array)
                    y_custom.append(digit)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

x_custom = np.array(x_custom).reshape(-1, 28, 28, 1)
y_custom = to_categorical(y_custom, 10)
print("Custom data loaded:", x_custom.shape)

# 3. Balance custom dataset (1000 samples per digit)
def balance_dataset(x, y, min_samples_per_class=1000):
    y_labels = np.argmax(y, axis=1)
    x_bal, y_bal = [], []
    for digit in range(10):
        idx = np.where(y_labels == digit)[0]
        count = len(idx)
        if count == 0:
            print(f"❗ No samples for digit {digit}")
            continue
        if count < min_samples_per_class:
            repeat = (min_samples_per_class // count) + 1
            idx = np.repeat(idx, repeat)[:min_samples_per_class]
        else:
            idx = idx[:min_samples_per_class]
        x_bal.append(x[idx])
        y_bal.append(y[idx])
    return np.concatenate(x_bal), np.concatenate(y_bal)

x_custom, y_custom = balance_dataset(x_custom, y_custom)
print("Balanced custom dataset:", x_custom.shape)

# 4. Combine MNIST and Custom
x_combined = np.concatenate([x_mnist, x_custom])
y_combined = np.concatenate([y_mnist, y_custom])
print("Combined dataset:", x_combined.shape)

# 5. Shuffle and Split for validation
x_train, x_val, y_train, y_val = train_test_split(
    x_combined, y_combined, test_size=0.1, stratify=np.argmax(y_combined, axis=1), random_state=42
)

# 6. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(x_train)

# 7. Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 8. Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# 9. Train
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, lr_schedule, checkpoint]
)

# 10. Evaluate on MNIST test
test_loss, test_acc = model.evaluate(x_test_mnist, y_test_mnist)
print(f"\n✅ Test Accuracy on MNIST: {test_acc:.4f}")

# 11. Per-digit accuracy
def evaluate_per_digit(x, y):
    preds = model.predict(x)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y, axis=1)
    for d in range(10):
        mask = true_labels == d
        acc = np.mean(pred_labels[mask] == d) if mask.sum() > 0 else 0
        print(f"Digit {d}: {acc:.4f}")

evaluate_per_digit(x_test_mnist, y_test_mnist)

# 12. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_history.png")
plt.close()

# 13. Save final model
model.save("mnist_plus_custom_model_final.keras")
print("\n✅ Final model saved as mnist_plus_custom_model_f.keras")

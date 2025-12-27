import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

FEATURE_DIR = "./mel_features"
CLASSES = ['Gazal', 'Lokdohori', 'Nephop', 'POP']

X, y = [], []

for label, genre in enumerate(CLASSES):
    genre_path = os.path.join(FEATURE_DIR, genre)
    print("Loading:", genre)

    for file in os.listdir(genre_path):
        if file.endswith(".npy"):
            mel = np.load(os.path.join(genre_path, file))
            if mel.shape == (128,128,1):   # sanity check
                X.append(mel)
                y.append(label)

X = np.array(X, dtype=np.float32)
y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

print("Dataset size:", X.shape, y.shape)

# Stratified split to preserve class balance
y_labels = np.argmax(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_labels
)

np.savez(
    "dataset.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

print("✅ Data loaded and saved to dataset.npz")
print("Train:", X_train.shape, " Test:", X_test.shape)

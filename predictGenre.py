import librosa
import numpy as np
import tensorflow as tf

# ----------------- SETTINGS -----------------
MODEL_PATH = "best_model.keras"
CLASSES = ["gazal", "lokdohori", "nephop", "pop"]

SR = 22050
DURATION = 30
SAMPLES = SR * DURATION
N_MELS = 128
TARGET_SHAPE = (128, 128)

# --------------------------------------------
# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

#yah ko sabai previous version jastai ho
def preprocess_audio(audio, sr, start, end):
    """Extract mel spectrogram from a slice of audio."""
    chunk = audio[start:end]
    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_img = tf.image.resize(mel_db[..., np.newaxis].astype(np.float32), TARGET_SHAPE)
    return np.expand_dims(mel_img, axis=0)

def predict_genre(file_path):
    audio, sr = librosa.load(file_path, sr=SR)
    length = len(audio)

    if length < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - length))
        length = len(audio)

    # Dataset ma pani 25%, 50% ra 75% use gareko le prediction ma use garda results accurate aayo!
    positions = [0.25, 0.5, 0.75]
    start_middle = (length - SAMPLES) // 2
    positions.append(start_middle / length)

    predictions = []
    for p in positions:
        start = int(p * length)
        end = start + SAMPLES
        if end <= length:
            processed = preprocess_audio(audio, sr, start, end)
            predictions.append(model.predict(processed)[0])

    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    index = np.argmax(avg_prediction)
    confidence = avg_prediction[index] * 100

    print("\n🎵 Predicted Genre:", CLASSES[index])
    print(f"Confidence: {confidence:.2f}%\n")


if __name__ == "__main__":
    test_file = input("Enter audio path to test (.wav): ")
    predict_genre(test_file)


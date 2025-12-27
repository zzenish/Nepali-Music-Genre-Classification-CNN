# Nepali-Music-Genre-Classification-CNN
Classifying Nepali music genres (POP, GAZAL, LOK DOHORI, NEPHOP) using CNN.
=======
# Nepali Music Genre Classification using CNN 🎶

This project applies **Convolutional Neural Networks (CNNs)** to classify Nepali music into four genres:
- Pop
- Gazal
- Lok Dohori
- Nephop

The model is trained on spectrograms generated from audio files, enabling it to learn frequency and temporal patterns unique to each genre.

---

## 📂 Project Structure

Repository (inside `cnn/`):

```
cnn/
├── datasets/            # Main dataset folder (not included in repo, ~94GB)
│   ├── Gazal/
│   ├── POP/
│   ├── Lokdohori/
│   └── Nephop/
├── audioProcessing.py    # Preprocessing: convert audio to spectrograms
├── splitDatasets.py      # Split dataset into train/test sets
├── trainModel.py         # Train CNN model
├── predictGenre.py       # Predict genre for new audio input
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/zzenish/Nepali-Music-Genre-Classification-CNN.git
cd Nepali-Music-Genre-Classification-CNN/cnn
```

2. Run run.sh

```bash
chmod +x run.sh
. run.sh
```

## 🧠 Model Architecture

Input: Spectrogram images of audio files

Typical layers used in this project:

- Convolutional layers + ReLU
- Max pooling layers
- Dropout (to reduce overfitting)
- Dense (fully connected) layers
- Output: Softmax with 4 classes (Pop, Gazal, Lok Dohori, Nephop)

## 🚀 Future Work

- Add more genres (e.g., Classical, Modern Rock)
- Experiment with deeper CNNs or transfer learning (ResNet, VGG)
- Deploy as a web app for real-time classification

## 📚 References

- [Stanford CS-230 — CNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [Librosa — audio and music analysis in Python](https://librosa.org/)
- [TensorFlow/Keras documentation](https://www.tensorflow.org/)
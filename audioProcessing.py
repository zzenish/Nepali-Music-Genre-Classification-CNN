import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Constants from Code 1
datasetDirectory = './datasets'
classes = ['Gazal', 'Lokdohori', 'Nephop', 'POP']
OUTPUT_DIR = "./mel_features"
SR = 22050
DURATION = 30
SAMPLES = SR * DURATION
N_MELS = 128

def loadAndPreprocessData(datasetDirectory, classes, target_shape=(128, 128)):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for classNumber, className in enumerate(classes):
        classDir = os.path.join(datasetDirectory, className)
        print("---------------Processing---------------")
        print(className)
        
        if not os.path.exists(classDir):
            continue

        for filename in os.listdir(classDir):
            if filename.endswith('.wav'):
                filepath = os.path.join(classDir, filename)
                
               
                audioData, sampleRate = librosa.load(filepath, sr=SR)

                # Extraction logic (Specific positions: 0.25, 0.5, 0.75)
                length = len(audioData) #It returns the total number of samples in the audio array, not the duration in seconds.
                positions = [0.25, 0.5, 0.75]
                
                for i,p in enumerate(positions):
                    start = int(p * length)
                    end = start + SAMPLES
                    
                    if end <= length:
                        chunk = audioData[start:end]
                        
                        # Mel Spectrogram with N_MELS from Code 1
                        melSpectrogram = librosa.feature.melspectrogram(
                            y=chunk, 
                            sr=sampleRate, 
                            n_mels=N_MELS
                        )
                        
                        # Convert to log scale (DB) as seen in Code 1
                        mel_db = librosa.power_to_db(melSpectrogram, ref=np.max)

                        # Resizing to (128, 128) and adding channel dimension 
                        mel_resized = tf.image.resize(
                            mel_db[..., np.newaxis].astype(np.float32),
                            target_shape 
                        ).numpy()
                        
                        base_name = os.path.splitext(filename)[0] 
                        save_filename = f"{base_name}_{i}.npy"
                        save_path=os.path.join(OUTPUT_DIR,className)
                        actual_savepath = os.path.join(save_path,save_filename)
                        os.makedirs(save_path, exist_ok=True)
                        np.save(actual_savepath,mel_resized)
                        
def main():
    loadAndPreprocessData(datasetDirectory, classes)
    
    
if __name__ == "__main__":
    main() # Call the main function
    print("\n🎉 Preprocessing complete! All Mel features saved to:", OUTPUT_DIR)


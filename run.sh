#!/bin/bash
set -e

echo "Creating virtual environment 'myenv'..."
python3 -m venv myenv

echo "Activating virtual environment..."
source myenv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Generating Random test data in testSample..."
python generateSampleAudio.py

echo "Splitting datasets for Training and Testing..."
python splitDatasets.py

echo "Starting model training..."
python trainModel.py

echo "Training complete!" 
echo "To test the model, run: python predictGenre.py" 
echo " Test audio files are in the testSample folder."

#python predictGenre.py

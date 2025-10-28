# EEG Jump Game

## Overview
This is a simple Pygame-based runner game controlled by EEG eye state predictions. The player must jump to avoid obstacles. The LSTM model predicts eye state (1 = eye closed → jump, 0 = eye open → no jump) from the EEG dataset.

## Dataset
- **Source:** [UCI Machine Learning Repository: EEG Eye State](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)
- **Instances:** 14 EEG channels + 1 target (eye state)
- **Format:** ARFF
- **Description:** Eye state was manually annotated via video analysis while EEG was recorded using Emotiv Neuroheadset.

## Requirements
- Python 3.10+
- Pygame
- TensorFlow / Keras
- NumPy, Pandas, scikit-learn, scipy

Install dependencies:
```bash
pip install -r requirements.txt

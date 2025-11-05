# UCI-HARUS-MLP

This project implements a Multi-Layer Perceptron (MLP) for Human Activity Recognition using the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

## Project Structure

- `dataset.py` — Contains the `UCIHAR` PyTorch dataset class for loading and preprocessing the UCI HAR data.
- `model.py` — Defines the MLP model architecture.
- `train.py` — Script for training the model.
- `evaluate.py` — Script for evaluating the trained model.
- `utils.py` — Utility functions for data handling and processing.
- `har_mlp.pth` — Saved PyTorch model weights.
- `requirements.txt` — Python dependencies.

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Download the UCI HAR Dataset:**
   Place the extracted dataset folder as `UCI HAR Dataset/` in the project root.

## Usage

- **Train the model:**
  ```sh
  python train.py
  ```

- **Evaluate the model:**
  ```sh
  python evaluate.py
  ```

## Dataset

The UCI HAR Dataset contains sensor data from smartphones for six human activities. See [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) for more details.
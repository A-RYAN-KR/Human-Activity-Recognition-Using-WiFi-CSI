# WiAR: Wi-Fi Based Human Activity Recognition

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a Deep Learning approach for Human Activity Recognition (HAR) using Channel State Information (CSI) from Wi-Fi signals. Utilizing the **WiAR Dataset**, the model combines **CNNs** for spatial feature extraction, **BiLSTMs** for temporal dependencies, and a custom **Self-Attention mechanism** to achieve high classification accuracy.

## 📊 Project Overview

Wi-Fi sensing allows for device-free activity recognition by analyzing how human movement distorts Wi-Fi signals (CSI). This project classifies **16 different human activities** (e.g., sitting, standing, walking, waving).

### Key Features
- **Data Processing:** Low Pass Filtering (Butterworth), Amplitude extraction, and Sequence padding/cropping.
- **Model Architecture:**
  - **CNN Blocks:** 1D Convolutions for feature extraction from subcarriers.
  - **Bi-Directional LSTM:** Captures temporal dynamics in both directions.
  - **Attention Layer:** A custom layer to weigh important time steps automatically.
- **Performance:** Achieved **~94% Accuracy** on the test set.

## 📂 Dataset

This project uses the **WiAR** dataset.
* **Source:** [GitHub - linteresa/WiAR](https://github.com/linteresa/WiAR)
* **Input Data:** CSI amplitude data extracted from `.dat` files.
* **Preprocessing:** The code automatically clones the dataset if running in a Colab environment. For local use, place data in the `data/` directory.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/WiAR-Activity-Recognition.git
   cd WiAR-Activity-Recognition

---

## Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
Note: csiread is required to parse Intel CSI format.


## 🚀 Usage
1. Training the Model
To train the model from scratch, run the training script (assuming you refactored the notebook, otherwise run the notebook directly):
```bash
python src/train.py
```

2. Using the Notebook
You can interactively explore the data and train the model using the provided notebook:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## 🧠 Model Architecture
The model follows this pipeline:
- **Input:** (Batch, 384, 180) - Fixed sequence length of 384 time steps.
- **Feature Extraction:** 3 layers of Conv1D + BatchNorm + MaxPooling.
- **Sequence Modeling:** 2 layers of Bi-Directional LSTMs.
- **Focus:** Custom Attention Layer.
- **Output:** Dense Layer (Softmax) for 16 classes.

## 📓 Google Colab

You can interactively explore the data and run the notebook in Google Colab:

[Open in Google Colab]([https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID_HERE](https://colab.research.google.com/drive/1ZFH5k0z5jHekHNPQVA1DxBTKny-FUVSn?usp=sharing))


## 📜 Citation
If you use this code or the WiAR dataset, please cite the original authors:
Guo, L., Wang, L., et al. "WiAR: A Public Dataset for Wi-Fi-based Activity Recognition."

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

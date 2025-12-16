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

### 4. How to Refactor Your Code (Briefly)

To look truly professional, don't just upload the notebook. Split the code:

1.  **`src/model.py`**: Put the `Attention` class and the `build_model` function here.
2.  **`src/data_loader.py`**: Put `load_and_preprocess_file`, `make_fixed_length_sequence`, and `augment_data` here.
3.  **`src/train.py`**: Import functions from the files above, run the data loading loop, and call `model.fit()`.

**If you don't want to refactor right now:**
1.  Clean up the Jupyter Notebook (remove the long output logs like "Loading files...").
2.  Save it inside the `notebooks/` folder.
3.  Upload the structure as described in Section 1.

### 5. Final Checklist Before Pushing

1.  **Remove Secrets:** Ensure no API keys or personal paths (like `/content/drive/`) are hardcoded (use relative paths).
2.  **Clean Notebook:** In Jupyter, go to `Cell > All Output > Clear` before saving to make the file smaller and cleaner.
3.  **Images:** Save the Confusion Matrix plot and Accuracy plot from your notebook as PNG files and put them in the `results/` folder so you can link them in the README.

**Git Commands to Upload:**
```bash
git init
git add .
git commit -m "Initial commit: WiAR Activity Recognition Project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
   

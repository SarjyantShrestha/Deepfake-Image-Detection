# A Solution to The Face DeepFake Detection Challenge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and model details for the paper **"A Solution to The Face DeepFake Detection Challenge,"** a project by students from Khwopa College of Engineering.

The project presents a lightweight and efficient deep-learning model capable of identifying manipulated facial images (deepfakes) with high accuracy. The model utilizes a **Convolutional Neural Network (CNN) architecture based on ResNet9** and was trained on a robustly augmented dataset to achieve state-of-the-art results.

## 🚀 Key Features

* **High Accuracy:** Achieves **93.8% accuracy** on the challenge's test dataset.
* **Top-Ranking Performance:** Outperforms other models in the comparative analysis, including those based on EfficientNet and Vision Transformers.
* **Lightweight Architecture:** Uses a ResNet9-based model with only **~6.6 million** total parameters, making it efficient.
* **Robust Training:** Trained on an augmented dataset of **50,000 images** (25,000 real, 25,000 fake) to handle common image alterations.
* **Excellent Class Separation:** Achieves an **Area Under ROC Curve (AUC) of 0.9796**.

## 📊 Performance

The model was evaluated on the challenge test set, which consisted of 7,000 images (5,000 fake and 2,000 real). This test set was specifically designed to test robustness by including alterations like rotation, mirroring, scaling, and recompression.

### Comparative Analysis

Our ResNet9-based model achieved the highest accuracy compared to other teams in the challenge.

| Rank | Team Name | Accuracy (%) |
| :--- | :--- | :---: |
| **-** | **ResNet9 (Ours)** | **93.80** |
| 1 | VisionLabs (EfficientNet) | 93.61 |
| 2 | Amped Team (EfficientNet) | 90.05 |
| 3 | AIMH Lab (Vision Transformer)| 72.62 |

### Overall Performance Metrics

| Class | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **Fake** | 96.08% | 0.952 | 0.960 | 0.955 |
| **Real** | 88.10% | 0.899 | 0.881 | 0.889 |
| **Macro Avg** | 92.09% | 0.925 | 0.920 | 0.922 |
| **Weighted Avg**| **93.80%** | **0.936** | **0.938** | **0.937** |

### Results Visualization

**Confusion Matrix**
| | **Predicted: Fake** | **Predicted: Real** |
| :--- | :---: | :---: |
| **Actual: Fake** | 4804 | 196 |
| **Actual: Real** | 238 | 1762 |

**Receiver Operating Characteristic (ROC) Curve**
The model shows excellent discrimination between Real and Fake classes, with an **AUC of 0.9796**.


## 🛠️ Model Architecture

The core of our solution is a custom Convolutional Neural Network (CNN) based on the **ResNet9** architecture. We utilize residual connections to allow the network to learn deeper features without suffering from vanishing gradients.

* **Total Parameters:** 6,591,106
* **Trainable Parameters:** 6,586,626
* **Non-trainable Parameters:** 4,480
* **Input Image Size:** $128 \times 128 \times 3$


## 💿 Dataset and Augmentation

* **Source:** The base dataset was from the **Face Deepfake Detection Challenge**, consisting of 15,000 real and fake images.
    * **Real Images (10,000):** Sourced from CelebA and FFHQ datasets.
    * **Fake Images (5,000):** Generated using models like StarGAN, AttGAN, StyleGAN, and StyleGAN2.
* **Augmentation:** To balance the dataset and improve robustness, the dataset was expanded to **50,000 total images** (25,000 real, 25,000 fake).
* **Transformations Applied:**
    1.  **Rotation:** Random angles from [45, 90, 135, 180, 225, 270, 315] degrees.
    2.  **Mirror:** Randomly flipped horizontally, vertically, or both.
    3.  **Scale:** Randomly scaled at a ratio between [0.5, 2.0].
    4.  **Compression:** Compressed using a JPEG quality factor between [50, 99].

## ⚙️ Training Configuration

The model was trained for 50 epochs using the following hyperparameters:

| Parameter | Value |
| :--- | :--- |
| **Learning Rate** | 0.001 |
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy Loss |
| **Epochs** | 50 |
| **Image Input Size** | $128 \times 128$ |


## 👨‍💻 Authors

* **Rupak Neupane**
* **Srijan Gyawali**
* **Sarjyant Shrestha**
* **Manish Pyakurel**

All authors are from the **Department of Computer and Electronics Engineering, Khwopa College of Engineering, IOE, Tribhuvan University**.

## 🙏 Acknowledgments

We would like to thank **Google Cloud Platform (GCP)** for providing free credits, which enabled us to utilize the NVIDIA A100 80GB GPU for the training process.

## 📜 How to Cite

If you use this work, please cite the original paper:

```bibtex
@inproceedings{neupane2024deepfake,
  title     = {A Solution to The Face DeepFake Detection Challenge},
  author    = {Rupak Neupane and Srijan Gyawali and Sarjyant Shrestha and Manish Pyakurel},
  booktitle = {IOE Graduate Conference},
  year      = {2024},
  publisher = {Khwopa College of Engineering, IOE, Tribhuvan University}
}

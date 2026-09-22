# RT-DETR Multi-Class Waste Detection

## 📌 Project Overview

This project presents a **Real-Time Detection Transformer (RT-DETR) based multi-class waste detection system** designed to identify different categories of waste from images.

The project focuses on applying a transformer-based object detection architecture for automated waste detection. The trained model can be used to perform inference on input images using the provided detection script and trained model weights.

The project was developed as part of an academic research project in **Computer Science & Engineering**.

---

## 🎯 Objectives

The main objectives of this project are:

* To develop a multi-class waste detection system using **RT-DETR**.
* To detect different types of waste objects from images.
* To utilize an end-to-end transformer-based object detection architecture.
* To train and evaluate the model using GPU-based computing resources.
* To provide a trained model that can be used for inference on new images.
* To support research into intelligent and automated waste detection systems.

---

## 🧠 Model Architecture

### RT-DETR

The project uses **RT-DETR (Real-Time Detection Transformer)** for multi-class object detection.

RT-DETR is an end-to-end object detection architecture based on the Transformer framework. It is designed to provide real-time object detection while maintaining competitive detection accuracy.

### Model Workflow

```text
Input Image
     ↓
RT-DETR Detection Model
     ↓
Feature Extraction & Transformer Processing
     ↓
Object Detection
     ↓
Bounding Boxes + Class Labels + Confidence Scores
     ↓
Detected Waste Objects
```

---

## ♻️ Multi-Class Waste Detection

The trained model is designed for detecting multiple categories of waste objects.

The project uses a multi-class waste dataset and trains RT-DETR to identify waste objects based on their visual characteristics.

The exact class labels used by the trained model are determined by the dataset configuration used during training.

---

## 📊 Model Training

The model was trained using **Kaggle GPU resources**.

### Training Platform

* **Platform:** Kaggle
* **Model:** RT-DETR
* **Task:** Multi-Class Object Detection
* **Framework:** PyTorch
* **Training:** GPU accelerated

The complete training process and notebook are available through the Kaggle resource provided below.

### Kaggle Training Notebook

🔗 **Kaggle Notebook:**
https://www.kaggle.com/code/lakshmirajravi/rt-detr-training-other

---

## 📁 Project Structure

The current project contains the following files:

```text
RT-DETR-MultiClass-Waste-Detection/
│
├── detect.py
├── model2.pt
├── README.md
└── .gitignore
```

### File Description

| File         | Description                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| `detect.py`  | Python script used for performing inference/detection using the trained model. |
| `model2.pt`  | Trained RT-DETR model weights.                                                 |
| `README.md`  | Project documentation and usage information.                                   |
| `.gitignore` | Specifies files and directories that should not be tracked by Git.             |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/lakshmiraj1426/RT-DETR-MultiClass-Waste-Detection.git
```

Navigate into the project directory:

```bash
cd RT-DETR-MultiClass-Waste-Detection
```

Install the required Python dependencies according to the environment used by the project.

A Python environment with the required RT-DETR/PyTorch dependencies is recommended for running the detection script.

---

## 🚀 Inference

The trained model is provided as:

```text
model2.pt
```

The detection script is:

```text
detect.py
```

To run the detection script:

```bash
python detect.py
```

Depending on the configuration inside `detect.py`, provide the required input image or input source to perform detection.

The model produces object detection results containing detected objects, class labels, bounding boxes, and confidence scores.

---

## 🔬 Research Application

This project is developed as part of research into **AI-based multi-class waste detection**.

The system demonstrates the application of Transformer-based object detection to waste-management-related computer vision tasks.

The project can serve as a foundation for further work involving:

* Automated waste detection
* Smart waste management
* Computer vision-based waste analysis
* Real-time object detection
* Transformer-based object detection
* Intelligent waste-management systems

---

## 📚 Project Resources

### GitHub Repository

🔗 https://github.com/lakshmiraj1426/RT-DETR-MultiClass-Waste-Detection

### Kaggle Training Notebook

🔗 https://www.kaggle.com/code/lakshmirajravi/rt-detr-training-other

---

## 👥 Authors

This project was developed by:

1. **Lakshmi Raj Ravi**
2. **Karnikula Mourya Mahesh**
3. **Nagaraju Shyam Vara Prasad Raju**
4. **Mustina Hima Kiran**
5. **Mohammad Amman Fawaz**
6. **K. V. Sambasiva Rao**

### Academic Details

**Program:** B.Tech – Computer Science & Engineering
**Institution:** DR. RVR NRI Institute of Technology Deemed to be University
**Batch:** 2023–2027

---

## 🏫 Institution

**DR. RVR NRI Institute of Technology Deemed to be University**

**Department of Computer Science & Engineering**

**Academic Batch: 2023–2027**

---

## 📌 Project Status

The trained RT-DETR model and inference code are provided in this repository.

The repository is intended for **academic, research, and educational purposes**.

---

## 📄 License

This project is intended for academic and research purposes. Please refer to the repository contents and associated research publication for further information regarding usage and attribution.

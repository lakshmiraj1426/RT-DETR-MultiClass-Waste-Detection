# RT-DETR-Based Multi-Class Waste Detection

## Overview

This repository provides the implementation of an RT-DETR-based multi-class
waste detection framework. It contains the dataset preparation workflow,
COCO-to-YOLO annotation conversion, RT-DETR training notebook, trained model
checkpoint, evaluation and inference procedures, and Streamlit-based
deployment.

The project is designed to provide a reproducible workflow from dataset
preparation and model training to model testing and practical image-based
inference.

---

## Research Work

**Title:**  
**RT-DETR-Based End-to-End Transformer Framework for Real-Time Multi-Class Waste Detection: Field Validation in Andhra Pradesh, India**

This repository contains the implementation and supporting materials associated
with the research work.

---

## Key Features

- RT-DETR-based multi-class waste detection
- 24 waste categories
- COCO annotation processing
- COCO-to-YOLO bounding-box conversion
- RT-DETR-L model training
- GPU-based training using Kaggle
- Model validation and evaluation
- Trained model checkpoint
- Image-based inference
- Streamlit application for interactive detection
- Kaggle training notebook
- Step-by-step execution and reproducibility guidance

---

## Project Workflow

```text
COCO Waste Dataset
        |
        v
Dataset Inspection
        |
        v
COCO Annotation Processing
        |
        v
COCO -> YOLO Annotation Conversion
        |
        v
24-Class YOLO Dataset
        |
        v
RT-DETR-L Training
        |
        v
Model Validation
        |
        v
Best Trained Model (best.pt)
        |
        +----------------------+
        |                      |
        v                      v
   Evaluation              Inference
        |                      |
        |                      v
        |                Detection Results
        |                      |
        +----------+-----------+
                   |
                   v
          Streamlit Application
```

---

## Waste Categories

The training configuration contains the following 24 waste categories:

1. Cardboard
2. Carton Packaging
3. Cigarette
4. Clean Paper
5. Clear Plastic
6. Contaminated Paper
7. Food Packaging
8. Food Scraps
9. Glass
10. Medical Waste
11. Metal
12. Paper Bag
13. Paper Cup
14. Plastic Bottle
15. Plastic Container
16. Plastic Cup
17. Plastic Lid
18. Plastic Packaging
19. Plastic Utensil
20. Printed Cardboard
21. Sanitary Waste
22. Straw
23. Styrofoam
24. Wood

---

## Dataset

The project uses a waste-detection dataset containing images and COCO-format
JSON annotations.

The original dataset contains `train`, `valid`, and `test` directories.
The current Kaggle training notebook converts the `train` and `valid`
annotations into YOLO format for RT-DETR training.

The complete dataset is not redistributed through this repository. Users
should obtain the dataset from its original source and follow the preparation
instructions described in this repository.

### Original Annotation Format

The source annotations are provided in COCO JSON format.

The conversion process:

```text
COCO JSON
    |
    v
Read image information
    |
    v
Read categories
    |
    v
Map valid categories to YOLO class IDs
    |
    v
Convert COCO bounding boxes
    |
    v
Generate YOLO .txt labels
```

---

## Repository Structure

```text
RT-DETR-MultiClass-Waste-Detection/
|
|-- README.md
|
|-- kaggle/
|   `-- RT-DETR_Waste_Detection_Kaggle.ipynb
|
|-- training/
|   |-- train.py
|   `-- coco_to_yolo.py
|
|-- evaluation/
|   `-- evaluate.py
|
|-- inference/
|   `-- inference.py
|
|-- streamlit/
|   `-- app.py
|
|-- models/
|   `-- best.pt
|
|-- configs/
|   `-- waste_data.yaml
|
|-- results/
|   |-- results.csv
|   |-- training_results.png
|   |-- F1_confidence_curve.png
|   |-- precision_confidence_curve.png
|   |-- precision_recall_curve.png
|   |-- recall_confidence_curve.png
|   |-- confusion_matrix.png
|   `-- confusion_matrix_normalized.png
|
|-- requirements.txt
|
`-- LICENSE
```

Only files that are actually included in the repository should be added to
the final repository structure.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Project implementation |
| PyTorch | Deep learning framework |
| Ultralytics | RT-DETR implementation |
| RT-DETR | Multi-class object detection |
| OpenCV | Image processing |
| Streamlit | Interactive deployment |
| Kaggle | GPU-based model training |

---

## Training Environment

The reported Kaggle training experiment used:

| Parameter | Configuration |
|---|---|
| Model | RT-DETR-L |
| Number of classes | 24 |
| Epochs | 150 |
| Image size | 640 × 640 |
| Batch size | 16 |
| GPU | NVIDIA Tesla T4 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Ultralytics | 8.4.90 |

The training run completed 150 epochs in approximately 6.76 hours.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/lakshmiraj1426/RT-DETR-MultiClass-Waste-Detection.git
cd RT-DETR-MultiClass-Waste-Detection
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the core packages:

```bash
pip install ultralytics streamlit opencv-python pillow
```

---

# Dataset Preparation

Place or mount the dataset according to the paths required by the training
scripts.

The original dataset used by the Kaggle notebook follows the general structure:

```text
dataset/
|
|-- train/
|   `-- _annotations.coco.json
|
|-- valid/
|   `-- _annotations.coco.json
|
`-- test/
    `-- _annotations.coco.json
```

The actual image files must be present in their corresponding directories.

---

# COCO-to-YOLO Conversion

The training notebook converts the COCO annotations into YOLO-format
bounding-box labels.

For each annotation, the COCO bounding box:

```text
[x, y, width, height]
```

is converted into the normalized YOLO representation:

```text
[class_id, x_center, y_center, width, height]
```

The converted dataset is organized as:

```text
waste_dataset/
|
|-- images/
|   |-- train/
|   `-- val/
|
`-- labels/
    |-- train/
    `-- val/
```

A dataset configuration file named:

```text
waste_data_converted.yaml
```

is generated for training.

---

# RT-DETR Training

The training model is loaded using the Ultralytics RT-DETR interface.

Example:

```python
from ultralytics import RTDETR

model = RTDETR("path/to/model.pt")
```

The reported training configuration is:

```text
Model:       RT-DETR-L
Epochs:      150
Image size:  640
Batch size:  16
Device:      GPU 0
Save:        True
Save period: 5 epochs
```

The training output is saved under the RT-DETR training results directory.

The best checkpoint is saved as:

```text
best.pt
```

---

# Kaggle Training Notebook

The complete Kaggle training workflow is provided in:

```text
kaggle/RT-DETR_Waste_Detection_Kaggle.ipynb
```

The notebook performs the following steps:

1. Configure the Kaggle environment.
2. Locate the dataset and model checkpoint.
3. Inspect the dataset.
4. Read COCO annotations.
5. Convert COCO annotations to YOLO format.
6. Create the RT-DETR dataset YAML file.
7. Load the RT-DETR model.
8. Train the model for 150 epochs.
9. Validate the model.
10. Save the trained model checkpoints.

The notebook can be opened directly in Kaggle or Jupyter Notebook.

---

# Model Evaluation

The trained model is evaluated using object-detection metrics:

- Precision
- Recall
- mAP@50
- mAP@50–95

## Final Epoch Results

The supplied `results.csv` records the following validation metrics at
epoch 150:

| Metric | Epoch 150 |
|---|---:|
| Precision | 0.94224 |
| Recall | 0.68448 |
| mAP@50 | 0.78248 |
| mAP@50–95 | 0.64636 |

## Best Recorded Values During Training

The highest recorded values across the 150 training epochs were:

| Metric | Best Value | Epoch |
|---|---:|---:|
| Precision | 0.94707 | 122 |
| Recall | 0.79020 | 80 |
| mAP@50 | 0.81431 | 90 |
| mAP@50–95 | 0.65688 | 81 |

The best values occur at different epochs and therefore should not be
interpreted as a single checkpoint's combined performance.

The validation metrics above are based on the supplied epoch-wise results.
They should not be interpreted as independent test-set results.

---

# Result Files

The repository contains the following result artifacts:

```text
results/
|
|-- results.csv
|-- training_results.png
|-- F1_confidence_curve.png
|-- precision_confidence_curve.png
|-- precision_recall_curve.png
|-- recall_confidence_curve.png
|-- confusion_matrix.png
`-- confusion_matrix_normalized.png
```

### Training Curves

`training_results.png` contains:

- Training GIoU loss
- Training classification loss
- Training L1 loss
- Validation GIoU loss
- Validation classification loss
- Validation L1 loss
- Precision
- Recall
- mAP@50
- mAP@50–95

### Confidence Curves

The repository includes:

- F1–Confidence curve
- Precision–Confidence curve
- Recall–Confidence curve

These curves show how the corresponding detection metrics vary with the
confidence threshold.

### Precision–Recall Curve

The Precision–Recall curve provides class-wise precision-recall behavior and
AP information across the evaluated classes.

### Confusion Matrices

Two confusion matrices are provided:

- `confusion_matrix.png` — raw detection counts
- `confusion_matrix_normalized.png` — normalized detection values

These provide class-level information about correct predictions and
classification errors.

---

# Testing

Testing should be performed using images that were not used for model training.

A typical testing workflow is:

```text
Test Image
    |
    v
Trained RT-DETR Model
    |
    v
Object Detection
    |
    +--> Bounding Box
    |
    +--> Predicted Class
    |
    `--> Confidence Score
```

The final repository should include the exact testing command used by the
project's testing script.

---

# Inference Using the Trained Model

The trained checkpoint can be used for image-based inference.

The final repository should contain the trained model at:

```text
models/best.pt
```

After placing the checkpoint in the required location, the inference script
can be used to process new images.

Example command format:

```bash
python inference/inference.py --model models/best.pt --source path/to/image.jpg
```

If the actual inference script uses different arguments, follow the command
documented in `inference/inference.py`.

The inference result contains the detected waste category, bounding box, and
confidence score.

---

# Streamlit Application

The project can be deployed through a Streamlit interface for interactive
image-based waste detection.

## Run the Application

From the project directory:

```bash
streamlit run streamlit/app.py
```

The application will provide a local web address, normally:

```text
http://localhost:8501
```

Open the address in a web browser.

## Application Workflow

```text
Upload Waste Image
        |
        v
Load RT-DETR Model
        |
        v
Run Detection
        |
        v
Display Bounding Boxes
        |
        v
Display Waste Classes
        |
        v
Display Confidence Scores
```

The exact Streamlit filename and model path should match the files included in
the repository.

---

# Step-by-Step Reproducibility Guide

To reproduce the reported workflow:

### Step 1 — Clone the repository

```bash
git clone https://github.com/lakshmiraj1426/RT-DETR-MultiClass-Waste-Detection.git
cd RT-DETR-MultiClass-Waste-Detection
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Obtain the dataset

Download the original waste dataset and place it in the required directory.

### Step 4 — Prepare annotations

Use the COCO annotation files supplied with the dataset.

### Step 5 — Convert annotations

Convert the COCO annotations into YOLO format using the supplied conversion
workflow.

### Step 6 — Configure the dataset

Create or use the generated YAML configuration containing the 24 waste
categories.

### Step 7 — Train RT-DETR

Use the supplied training script or Kaggle notebook.

The reported experiment used:

```text
150 epochs
640 image size
batch size 16
NVIDIA Tesla T4
RT-DETR-L
```

### Step 8 — Save the best model

The trained checkpoint is saved as:

```text
best.pt
```

### Step 9 — Evaluate the model

Run the evaluation workflow and record the detection metrics.

### Step 10 — Test the model

Use independent test images to evaluate detection performance.

### Step 11 — Run inference

Use the trained `best.pt` checkpoint on new images.

### Step 12 — Run Streamlit

```bash
streamlit run streamlit/app.py
```

---

# Troubleshooting

## Model Not Found

Check that the trained checkpoint exists:

```text
models/best.pt
```

Also verify that the path used in the inference or Streamlit code matches the
actual location of the model.

## ModuleNotFoundError

Activate the virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

## CUDA / GPU Problems

Confirm that PyTorch can access the GPU when GPU training is required.

For Kaggle training, enable a GPU accelerator before running the notebook.

## No Detections

Try the following:

- Check that the correct trained checkpoint is loaded.
- Verify that the input image contains objects represented in the training
  classes.
- Check the confidence threshold.
- Verify the image format.
- Confirm that the model and dataset configuration correspond to the same
  class mapping.

## Streamlit Port Already in Use

Run Streamlit using another port:

```bash
streamlit run streamlit/app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

# Limitations

- The complete dataset is not redistributed through this repository.
- Model performance depends on the training dataset and image conditions.
- The current training notebook converts the training and validation splits
  for RT-DETR training.
- The reported validation metrics are not independent test-set metrics.
- Real-world images can contain variations in lighting, object scale,
  background, occlusion, and object condition.
- The Streamlit interface is intended for image-based inference.

---

# Code Availability

The repository provides the source code and supporting materials required to
understand and reproduce the RT-DETR-based multi-class waste detection
workflow, including the Kaggle training notebook, dataset-conversion
procedure, trained model checkpoint, inference workflow, evaluation
procedure, result files, and Streamlit deployment.

**Repository:**  
https://github.com/lakshmiraj1426/RT-DETR-MultiClass-Waste-Detection

---

# Authors

This project was developed by:

Nagaraju Shyam Vara Prasad Raju

Karnikula Mourya Mahesh

Mustina Hima Kiran

Mohammad Amman Fawaz

Lakshmi Raj Ravi

K. V. Sambasiva Rao

Program: B.Tech – Computer Science & Engineering

Institution: DR. RVR NRI Institute of Technology Deemed to be University

Batch: 2023–2027

# Citation

If you use this repository or the associated methodology in academic research,
please cite the corresponding research paper.

```text
Citation will be added after publication.
```

---

# License

A license should be added to the repository if the project is intended for
public reuse. The README should be updated to match the actual `LICENSE` file
included in the repository.

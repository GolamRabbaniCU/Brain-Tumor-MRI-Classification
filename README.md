# Brain Tumor MRI Classifier

## Overview
This project fine-tunes a Vision Transformer (ViT) model to classify brain MRI scans into
four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. The model
was trained on Kaggle using a T4 GPU, and predictions are served locally through a FastAPI
backend running on CPU.

## Model
- **Architecture:** `google/vit-base-patch16-224` (fine-tuned)
- **Dataset:** Brain Tumor MRI Dataset (Kaggle -- `masoudnickparvar/brain-tumor-mri-dataset`)
- **Classes:** glioma, meningioma, notumor, pituitary
- **Test Accuracy:** 94.37% (0.9437)
- **Epochs:** 6

### Per-Epoch Results

| Epoch | Training Loss | Test Accuracy |
|-------|---------------|----------------|
| 1     | 0.5758        | 0.8938         |
| 2     | 0.0764        | 0.9375         |
| 3     | 0.0158        | 0.9425         |
| 4     | 0.0066        | 0.9487         |
| 5     | 0.0038        | 0.9463         |
| 6     | 0.0030        | 0.9437         |

## API Endpoints

| Method | Endpoint  | Description                              |
|--------|-----------|-------------------------------------------|
| GET    | /health   | Server status and model info               |
| POST   | /predict  | Upload MRI image, get prediction            |

## Prediction Response
```json
{
  "label": "glioma",
  "confidence": 0.9821
}
```

## Screenshots

### Swagger UI
![Swagger](screenshots/swagger.png)

## Installation
```bash
git clone https://github.com/GolamRabbaniCU/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
pip install -r requirements.txt
```

## Run
```bash
fastapi dev main.py
# Open http://localhost:8000/docs
```

## Technologies Used
- Python
- PyTorch, Hugging Face Transformers
- FastAPI
- Pillow

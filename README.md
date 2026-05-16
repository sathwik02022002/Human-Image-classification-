# Human Image Classification using Wavelet Features and Classical ML

End-to-end human facial image classification pipeline using OpenCV face detection, 2D Haar wavelet feature engineering, and multi-model experimentation with GridSearchCV hyperparameter tuning.

---

## What It Does

Given a dataset of human face images organized by category, the pipeline:

1. Detects and crops faces using OpenCV Haar Cascade classifiers (requires both eyes to be detected — strict quality filter)
2. Extracts a combined feature vector from raw pixel data and Daubechies wavelet-transformed image
3. Trains and benchmarks multiple classifiers with GridSearchCV
4. Saves the best model and runs inference on new images, auto-sorting predictions into output folders

---

## Pipeline Overview

```
Raw Images
    → OpenCV face detection (haarcascade_frontalface_default.xml)
    → Eye detection filter (≥2 eyes required for crop to proceed)
    → Dual feature extraction:
        • Raw image: resized to 32×32 → flattened (3072 features)
        • Wavelet transform: PyWavelets db1 level-5 decomposition → resized 32×32 → flattened (1024 features)
        • Combined feature vector: 4096 dimensions
    → Train/test split (80/20, random_state=0)
    → StandardScaler normalization
    → Model benchmarking with GridSearchCV (5-fold CV)
    → Best model serialized with pickle
    → Inference: predict category and auto-move image to output folder
```

---

## Models Benchmarked

| Model | Hyperparameter Search Space |
|---|---|
| SVM (RBF kernel) | C ∈ {1, 10, 100, 1000}, kernel ∈ {rbf, linear} |
| Random Forest | n_estimators ∈ {1, 5, 10} |
| Logistic Regression | C ∈ {1, 5, 10}, solver=liblinear |

All models wrapped in `sklearn.pipeline.make_pipeline` with `StandardScaler` for consistent normalization. GridSearchCV selects best estimator per model; scores are compared across models.

---

## Feature Engineering Detail

**Why wavelets?**

Raw pixel features alone are sensitive to lighting and pose. Haar wavelet decomposition (via PyWavelets `db1` at level 5) captures edge and texture frequency information that is more stable across illumination changes. The approach zeroes out the approximation coefficients (`coeffs_H[0] *= 0`) and reconstructs only from detail coefficients — isolating structural edge features while suppressing low-frequency lighting variation.

**Combined feature vector construction:**

```python
# Raw: (32, 32, 3) → flatten → 3072
scalled_raw_img = cv2.resize(img, (32, 32))

# Wavelet: grayscale → db1 level-5 decompose → zero approx → reconstruct → (32, 32) → flatten → 1024
img_har = w2d(img, 'db1', 5)
scalled_img_har = cv2.resize(img_har, (32, 32))

# Stack vertically → 4096-dimensional feature vector
combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
```

---

## Tech Stack

- **Python 3.x**
- **OpenCV 4.5.1** — face and eye detection via Haar Cascades
- **PyWavelets 1.1.1** — Daubechies wavelet (`db1`) decomposition
- **scikit-learn** — SVM, Random Forest, Logistic Regression, GridSearchCV, Pipeline, StandardScaler, classification report
- **NumPy** — feature vector construction
- **Matplotlib / Seaborn** — visualization
- **pickle** — model serialization

---

## Project Structure

```
Human-Image-classification-/
├── Human Faces Major Project B1/
│   └── model/
│       ├── Major project ML-B1.ipynb   # Full pipeline notebook
│       ├── human images model.p        # Serialized best model (pickle)
│       ├── requirements.txt
│       └── opencv/
│           └── haarcascades/           # OpenCV XML classifier files
│               ├── haarcascade_frontalface_default.xml
│               ├── haarcascade_eye.xml
│               └── (13 additional cascade files)
└── README.md
```

---

## Setup

```bash
git clone https://github.com/sathwik02022002/Human-Image-classification-
cd "Human-Image-classification-/Human Faces Major Project B1/model"
pip install -r requirements.txt
jupyter notebook "Major project ML-B1.ipynb"
```

**Dataset structure expected:**

```
humandataset/
├── Indian faces/
│   ├── img1.jpg
│   └── ...
└── Other country faces/
    ├── img1.jpg
    └── ...
```

---

## Inference

After training, the saved model (`human images model.p`) accepts new images via interactive input, predicts the category, and automatically moves the file to the corresponding output folder:

```python
model = pickle.load(open('human images model.p', 'rb'))
# Prompts for image paths → predicts → sorts into Indian faces / Other country faces folders
```

---

## Key Design Decisions

- **Two-eye filter**: Only images where OpenCV detects ≥2 eyes pass preprocessing. This enforces data quality by rejecting profile views, occluded faces, and low-resolution crops before feature extraction.
- **Wavelet approximation zeroing**: Removing low-frequency approximation coefficients before reconstruction isolates structural detail features, reducing sensitivity to global illumination variation.
- **Pipeline-based grid search**: Using `sklearn.pipeline.make_pipeline` ensures the `StandardScaler` is fit only on training data during each cross-validation fold — preventing data leakage.

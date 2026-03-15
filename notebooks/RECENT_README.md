# Ground-Based Cloud Image Classification

A computer vision project for classifying cloud types from ground-based sky photographs, using the NASA GLOBE Observer dataset. The goal is to replicate and extend findings from an academic paper on cloud classification using classical machine learning pipelines.

---

## Project Status

**Current notebook:** `colab_05_eda_visualize_insights_replicate_paper_with_smote.ipynb`

This is the 5th major iteration. It focuses on exploratory data analysis, class imbalance correction via image augmentation (with SMOTE in the name as a conceptual reference), and a baseline multi-classifier evaluation sweep using flattened grayscale image features.

---

## Dataset

**Source:** NASA GLOBE Cloud Observer dataset
**Location:** `../resources/cloud-images/NASA_GLOBE_CD/downloaded_images/`

Images are organized into subdirectories by cloud label. The loader caps each class at 3,000 images to avoid memory issues during experimentation.

**Cloud classes encountered in the data (abbreviated labels):**
- `ClSk` — Clear Sky
- `Ct` — Contrails
- Several additional cloud-type categories (cumulus, stratus, cirrus variants, etc.)

The dataset has a notable **class imbalance**, with `ClSk` (Clear Sky) and `Ct` (Contrails) being significantly underrepresented compared to other cloud types.

---

## Notebook Walkthrough

### 1. Image Indexing
The notebook scans the image directory, builds a dictionary mapping each filename to its label and path, and caps each class at 3,000 samples.

### 2. Image Loading & Preprocessing
Images are loaded with OpenCV, converted to **grayscale**, and resized to **64×64 pixels**. Labels are extracted in parallel with the images. Failed/missing reads are skipped gracefully.

### 3. Exploratory Data Analysis
Two sets of bar charts are produced:
- **Proportional distribution** of labels across the full dataset
- **Absolute count distribution** to surface the raw imbalance

### 4. Train/Test Split
A `StratifiedShuffleSplit` (80/20) preserves class proportions across the split. Labels are integer-encoded with `LabelEncoder`. A side-by-side bar chart confirms the split is stratified correctly.

### 5. Data Augmentation for Class Balancing
Rather than SMOTE (which is referenced in the filename conceptually), the notebook uses **torchvision-based augmentation** to oversample the two minority classes (`ClSk` at index 5, `Ct` at index 7).

The augmentation pipeline randomly applies combinations of:
- **Wrap translation** (circular roll) and **border translation** (black-padded affine shift), up to ±14% of image dimensions
- Horizontal flip (p=0.5) and vertical flip (p=0.15)
- Random rotation (±20°)
- Random scaling (90–110%) and random resized crop (80–100% of area)

Two pipeline modes — `stacked` (applies multiple transforms in sequence) and `one_of` (applies one at random) — are randomly chosen per image. Minority class images are duplicated until they approach the count of a majority class (capped at 80% × 3,000 = 2,400 samples per class).

After augmentation, the combined training set is **shuffled** to prevent ordering artifacts.

### 6. Feature Extraction
Images are **flattened** from `(64, 64)` to `4096`-dimensional vectors for use with scikit-learn classifiers. No CNN or learned feature extraction is used at this stage — raw pixel values are the feature set.

### 7. Baseline Classifier Sweep
A suite of classifiers is evaluated with **5-fold stratified cross-validation**, scored by **macro F1** (appropriate for imbalanced multi-class problems). Models include:

| Model | Notes |
|---|---|
| Gaussian Naive Bayes | No scaling |
| Logistic Regression | With StandardScaler |
| Linear SVC (SGD) | With StandardScaler |
| SVC (RBF kernel) | With StandardScaler, `probability=True` |
| K-Nearest Neighbors | With StandardScaler |
| Decision Tree | No scaling |
| Random Forest | No scaling |
| Gradient Boosting | 150 estimators, lr=0.1, max_depth=3, subsample=0.7 |
| HistGradientBoosting | No scaling |
| MLP (Neural Net) | With StandardScaler, max_iter=500 |

Results are returned as a DataFrame sorted by mean cross-val score, with standard deviation and wall-clock time per model.

---

## Key Design Decisions

- **Grayscale only:** Color information is discarded. This is intentional for the baseline — cloud texture and structure are the discriminating features, not color.
- **64×64 resolution:** Balances detail retention with memory and compute constraints for classical ML on flat features.
- **Augmentation over SMOTE:** SMOTE on raw pixel vectors tends to produce blurry, unrealistic interpolations. Augmentation generates more plausible synthetic samples. The filename references SMOTE as the original plan.
- **Macro F1 as the metric:** Treats all classes equally regardless of size, which is appropriate given the imbalance.

---

## Dependencies

```
numpy
pandas
matplotlib
opencv-python (cv2)
torch
torchvision
scikit-learn
tqdm
joblib
Pillow
```

---

## Project Structure

```
project-root/
├── resources/
│   └── cloud-images/
│       └── NASA_GLOBE_CD/
│           └── downloaded_images/
│               ├── ClSk/
│               ├── Ct/
│               └── <other cloud type dirs>/
└── notebooks/
    └── colab_05_eda_visualize_insights_replicate_paper_with_smote.ipynb
```

---

## Next Steps / Open Questions

- [ ] Evaluate which classifier(s) from the sweep perform best and move to hyperparameter tuning
- [ ] Experiment with HOG or LBP features instead of raw pixels
- [ ] Investigate whether color channels (HSV or LAB) improve performance
- [ ] Consider a lightweight CNN baseline (e.g., a small ResNet or MobileNet fine-tune) to compare against classical models
- [ ] Revisit SMOTE on feature-extracted representations (rather than raw pixels) if augmentation coverage is still insufficient
- [ ] Confirm which paper is being replicated and align metric reporting to match paper's evaluation protocol

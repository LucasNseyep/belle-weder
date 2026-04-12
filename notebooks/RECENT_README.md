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
- [ ] **Pretrained ResNet embeddings as features:** Load a pretrained ResNet18, strip the final FC layer, and pass all images through it to obtain 512-dimensional embeddings per image. Feed these into the existing `evaluate_classifiers` sweep instead of raw flattened pixels. This replaces 16,384 noisy pixel features with 512 rich visual features (edges, textures, shapes) learned from ImageNet — expected to boost all classifiers in the sweep, especially Logistic Regression. No fine-tuning required; the backbone is frozen and used purely for feature extraction.

---

# Checkpoint 2 — Transfer Learning & NASA GLOBE Data Cleaning

**Current notebooks:** see table below

This checkpoint covers the move from classical ML on flat pixels to transfer learning with ResNet18, and a parallel effort to clean the dirty NASA GLOBE dataset using a trained binary classifier.

---

## Why NASA GLOBE Images Are Dirty

The NASA GLOBE Observer app asks volunteers to photograph clouds. However, it instructs users to photograph all four cardinal directions (N, E, S, W) plus zenith and nadir — not just the sky. This means a large fraction of images in each labeled folder (e.g. `Ac/`, `Cu/`) actually show walls, roads, trees, and ground-level scenes rather than clouds. These images cannot be used for training as-is.

---

## Notebooks Overview

| Notebook | Purpose |
|---|---|
| `remove_non_cloud_images.ipynb` | Binary cloud/clear_sky classifier sweep with augmentation-leak fix |
| `fine_tuning_rf_mlp.ipynb` | Hyperparameter search for RF and MLP without cross-validation |
| `fine_tuning_rf_mlp_cv.ipynb` | Same search using `RandomizedSearchCV` with `AugmentingClassifier` |
| `resnet18_prediction_distribution.ipynb` | EDA: what does fully pretrained ResNet18 predict on cloud images? |
| `resnet18_imagenet_probs_rf_cloud_classifier.ipynb` | Option C transfer learning: 1000-dim softmax vectors → Random Forest |
| `resnet18_multihead_512_cloud_classifier.ipynb` | Multi-head ResNet18: frozen backbone + new 2-class head trained from scratch |
| `resnet18_cloud_filter_nasa_ac.ipynb` | Apply trained model to filter the NASA GLOBE `Ac` folder |
| `resnet18_cloud_filter_all_classes.ipynb` | Apply trained model to all NASA GLOBE folders, export combined CSV |

---

## 1. Binary Cloud / Clear Sky Classifier (`remove_non_cloud_images.ipynb`)

A small, clean dataset (`cloud-classifier-1/`) was assembled with two classes — `cloud` and `clear_sky` — to train a binary filter for the dirty NASA GLOBE images.

### Augmentation Leak Fix

The original notebook augmented data before passing it to `cross_val_score`, meaning augmented versions of the same image could appear in both the training fold and the validation fold — inflating scores artificially. This was fixed with an `AugmentingClassifier` wrapper that applies augmentation only inside `fit()`:

```python
class AugmentingClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X_aug, y_aug = self._augment(X, y)          # augment here, inside the fold
        self.base_clf.fit(X_aug.reshape(len(X_aug), -1), y_aug)
    def predict(self, X):
        X_norm = X.astype(np.float32) / 255.0        # same normalisation as training
        return self.base_clf.predict(X_norm.reshape(len(X_norm), -1))
```

`evaluate_classifiers` wraps every classifier with `AugmentingClassifier` before passing it to `cross_val_score`, so all models benefit from the fix automatically.

### Results

| Split | Accuracy | F1 macro |
|---|---|---|
| Cross-val (5-fold) | ~0.928 | ~0.928 |
| Held-out test set (140 images) | 0.9429 | 0.9437 |

Random Forest and MLP were the top two classifiers.

---

## 2. Hyperparameter Tuning (`fine_tuning_rf_mlp.ipynb` / `fine_tuning_rf_mlp_cv.ipynb`)

Two tuning notebooks were created in parallel:

- **`fine_tuning_rf_mlp.ipynb`** — uses `ParameterSampler` (no CV). Fits on pre-augmented training data and evaluates directly on the test set. Faster on a MacBook; good for quick iteration.
- **`fine_tuning_rf_mlp_cv.ipynb`** — uses `RandomizedSearchCV` with `AugmentingClassifier`. Parameters are prefixed `base_clf__` to thread through the wrapper. `n_jobs=1` is set because PyTorch's MPS augmentation is not safe under multiprocessing.

Random Forest parameter space (MacBook-friendly):
```python
rf_param_dist = {
    "n_estimators":           randint(50, 150),
    "max_depth":              [5, 10, 15, 20],
    "min_samples_split":      randint(2, 15),
    "min_samples_leaf":       randint(1, 8),
    "max_features":           ["sqrt", "log2"],
    "criterion":              ["gini", "entropy"],
    "bootstrap":              [True, False],
    "min_impurity_decrease":  [0.0, 0.01, 0.05],
}
```

---

## 3. ResNet18 Prediction Distribution EDA (`resnet18_prediction_distribution.ipynb`)

Before training any new heads, all 698 images from `cloud-classifier-1/` were passed through a fully pretrained ResNet18 (ImageNet 1000-class head intact) to understand what the network natively sees.

**Key finding:** ResNet18 never predicts "cloud" because ImageNet has no cloud class. For sky photos it tends to predict `sandbar`, `seashore`, `beacon`, and `geyser` — classes that share the same wide-sky + flat-horizon visual structure as a sky photograph. This confirmed that the ImageNet head is useless for our domain, but the backbone features it produces are rich and exploitable.

Three visualisations:
1. Top-15 predicted ImageNet classes per true label (bar chart)
2. Sample images with top-3 predictions overlaid
3. Top-1 confidence distribution per category

---

## 4. Transfer Learning Approaches

Two architectures were evaluated for binary cloud/clear_sky classification using the pretrained ResNet18 backbone.

### Option C — 1000-dim Softmax Features → Random Forest (`resnet18_imagenet_probs_rf_cloud_classifier.ipynb`)

Images are passed through the full pretrained ResNet18 (head intact). The 1000-dim softmax probability vector for each image is used as the feature vector. A Random Forest is trained on top — no ResNet weights are modified.

**Result: 97.1% accuracy, 0.971 F1 macro** on 140 held-out images.

Feature importance analysis revealed which ImageNet classes drive the prediction (e.g. sky-adjacent classes like `seashore`, `sandbar` signal cloud; more structured object classes signal clear_sky).

### Multi-Head ResNet18 (`resnet18_multihead_512_cloud_classifier.ipynb`)

A second linear head (`head2`) is added on top of the 512-dimensional penultimate backbone features. The backbone and the original ImageNet head (`head1`) are both frozen. Only `head2` (1,026 parameters out of ~11 million total) is trained.

```
backbone (frozen) → 512-dim features → head1 (frozen, 1000 classes — ImageNet)
                                      → head2 (trained,    2 classes — cloud/clear_sky)
```

**Why this works so much better than 12-class cloud type classification:** The binary task (cloud vs no-cloud) maps cleanly onto existing backbone features. The backbone already encodes sky texture, horizon lines, and diffuse luminance from ImageNet training — all of which are sufficient to separate sky from non-sky. Distinguishing 12 fine-grained cloud types requires much subtler texture discrimination that the frozen backbone can't provide without fine-tuning.

**Result after 50 epochs: 100% accuracy, 1.000 F1 macro** on 140 held-out images (genuine — not overfitting; the binary task is simply well-separated in 512-dim feature space).

The trained model is saved to `models/multihead_resnet18_cloud_sky.pth`:
```python
torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "classes":              [str(c) for c in ordinal_encoder.classes_],
    "num_classes":          len(ordinal_encoder.classes_),
}, SAVE_PATH)
```

**Note on PyTorch 2.6:** `torch.load` now defaults to `weights_only=True`. Loading checkpoints that contain numpy strings requires `weights_only=False`. Classes are saved as plain Python strings (`str(c)`) rather than numpy strings to avoid this in future.

---

## 5. NASA GLOBE Data Cleaning

### Ac Folder Filter (`resnet18_cloud_filter_nasa_ac.ipynb`)

The trained multi-head model (head2) is applied to all 35,627 images in the `Ac/` folder. Results:

| | Count | % |
|---|---|---|
| Predicted cloud | 25,679 | 72.1% |
| Predicted not cloud | 9,948 | 27.9% |
| Kept (conf ≥ 0.8) | 22,492 | 63.1% |

The notebook provides three visualisations:
1. **Confidence distribution** — histogram of head2 cloud scores across all images
2. **Not-cloud samples** — 4×4 grid of the lowest-confidence images (ground, trees, walls)
3. **Borderline samples (0.5–0.8 confidence)** — 4×4 grid showing uncertain images with both head2 cloud score and head1 top ImageNet prediction per image

Clean filenames are exported to `ac_clean_filenames.txt`.

**On head1 labels for borderline images:** Head1 (the ImageNet head) never labels these images as "tree" even when trees are visible, because ImageNet has no generic tree class — only hyper-specific species (`oak`, `palm`, `fig`) that only activate when the tree is the centred, dominant subject. For sky photos with edge-of-frame vegetation, the dominant visual signal is still sky, so head1 falls back to outdoor-horizon classes (`sandbar`, `seashore`, `beacon`).

### All-Classes Filter (`resnet18_cloud_filter_all_classes.ipynb`)

A single notebook that loops over all 11 cloud-type folders (skipping `ClSk` — clear sky is already the negative class the model was trained on), runs batch inference, and collects everything into one combined DataFrame:

| Column | Description |
|---|---|
| `folder` | Cloud type label (e.g. `Cu`, `Ns`) |
| `filename` | Image filename |
| `cloud_conf` | head2 probability of cloud |
| `head2_pred` | `cloud` or `clear_sky` |
| `head1_class` | Top-1 ImageNet class from head1 |
| `head1_conf` | head1 top-1 confidence |

Results are exported to a single CSV: `resources/cloud-images/NASA_GLOBE_CD/cloud_filter_results.csv`. This CSV is the reference for downstream training — join on `folder + filename` and filter to `cloud_conf >= 0.8` to get a clean subset of any class.

A bar chart shows the keep-rate per folder, making it easy to spot which cloud types have the most non-cloud contamination.

---

## Key Design Decisions

- **Binary filter before multi-class training:** Cleaning the dataset first means the multi-class model is never exposed to ground/wall/tree images labelled as cloud types.
- **CSV reference over file copying:** The filter results are stored as a CSV pointer rather than copying images to a new directory. This is cheaper on disk and fully reversible.
- **Multi-head over full fine-tuning:** Only 1,026 parameters are trained. This avoids overfitting on 698 labelled images and runs in minutes on Apple Silicon MPS.
- **0.8 confidence threshold:** Chosen to be conservative — prefer a smaller clean dataset over including borderline images that might be ambiguous.
- **Skip `ClSk` in the all-classes filter:** Clear sky is already the negative class head2 was trained on; running the filter over it would be circular.

---

## Project Structure (updated)

```
project-root/
├── models/
│   └── multihead_resnet18_cloud_sky.pth       ← trained binary classifier
├── resources/
│   └── cloud-images/
│       ├── cloud-classifier-1/                ← clean labelled set (698 images)
│       │   ├── clear_sky/
│       │   └── cloud/
│       └── NASA_GLOBE_CD/
│           ├── downloaded_images/
│           │   ├── Ac/  As/  Cb/  Cc/  Ci/
│           │   ├── ClSk/  Cs/  Ct/  Cu/
│           │   └── Ns/  Sc/  St/
│           ├── ac_clean_filenames.txt          ← Ac-only filter output
│           └── cloud_filter_results.csv        ← all-classes filter output
└── notebooks/
    ├── remove_non_cloud_images.ipynb
    ├── fine_tuning_rf_mlp.ipynb
    ├── fine_tuning_rf_mlp_cv.ipynb
    ├── resnet18_prediction_distribution.ipynb
    ├── resnet18_imagenet_probs_rf_cloud_classifier.ipynb
    ├── resnet18_multihead_512_cloud_classifier.ipynb
    ├── resnet18_cloud_filter_nasa_ac.ipynb
    └── resnet18_cloud_filter_all_classes.ipynb
```

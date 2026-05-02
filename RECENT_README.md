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
    ├── resnet18_cloud_filter_all_classes.ipynb
    ├── resnet18_cloud_filter_analysis.ipynb
    ├── 07_transfer_learning.ipynb
    └── 08_transfer_learning_filtered.ipynb
```

---

# Checkpoint 3 — ViT Feature Embeddings for 11-Class Cloud Classification

**Current notebooks:** `07_transfer_learning.ipynb`, `08_transfer_learning_filtered.ipynb`

This checkpoint covers the attempt at fine-grained 11-class cloud type classification using a Vision Transformer (ViT-B-16), first on the raw NASA GLOBE images, then repeated on the dataset cleaned by the ResNet18 filter from Checkpoint 2.

---

## Model: ViT-B-16 with SWAG Weights

`vit_b_16` from torchvision loaded with `IMAGENET1K_SWAG_E2E_V1` weights — a stronger pretrained checkpoint than the standard ImageNet-1K weights, trained with weakly-supervised data. Input size is fixed at **384×384** by these weights.

The ImageNet classification head is replaced with a new head:
```python
model.heads = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(768, 11))
```

The entire encoder is frozen. Only the 11-class head is trained (~8,459 parameters out of ~86 million total).

---

## Feature Embedding and Caching

Rather than running full forward passes through the ViT encoder on every training epoch, features are extracted once from the frozen encoder and cached to disk. Training then operates entirely on the cached embeddings.

The class token output (`x[:, 0]`) from the ViT encoder is used as the 768-dimensional embedding for each image:

```python
def extract_features(model, loader, device):
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            x = model._process_input(imgs)
            batch_class_token = model.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = model.encoder(x)
            features = x[:, 0]   # 768-dim class token output
```

Cached splits are saved as `cached_features/train.pt`, `val.pt`, `test.pt` (notebook 07) and `cached_features_filtered/` (notebook 08). The cache is only recomputed if the files don't exist.

**Why cache?** Feature extraction through ViT at 384×384 is slow (~minutes per epoch if done inline). Caching reduces each training epoch to a few seconds of linear head forward+backward passes on 768-dim vectors, making 300-epoch runs practical on a MacBook.

---

## Data Pipeline

- **Split:** 80% train / 10% val / 10% test, stratified by class (`StratifiedShuffleSplit`)
- **Max per class:** 3,000 images
- **Class imbalance:** handled with `WeightedRandomSampler` — each class is sampled proportionally so the head sees a balanced stream regardless of raw class sizes
- **Augmentation:** applied during feature extraction (train split only), including wrap/border translation, random flips, rotation, scale, and resized crop
- **Classes:** 11 cloud types (Ac, As, Cb, Cc, Ci, Cs, Ct, Cu, Ns, Sc, St) — `ClSk` excluded

---

## Training

- **Optimiser:** AdamW (default learning rate)
- **Loss:** CrossEntropyLoss
- **Epochs:** up to 300 with early stopping (patience=100)
- **Checkpoint:** best validation accuracy saved to `best_model.pt` / `best_model_filtered.pt`

---

## Results

| Notebook | Dataset | Test accuracy |
|---|---|---|
| `07_transfer_learning.ipynb` | Raw NASA GLOBE (dirty) | **47.5%** |
| `08_transfer_learning_filtered.ipynb` | Filtered via ResNet18 CSV | TBD |

The 47.5% baseline on dirty data is the reference. The filtered run (notebook 08) is the first attempt to quantify how much the data cleaning improves the result.

---

## Why 47.5% Is Hard to Beat with a Frozen Encoder

Two factors compound here:

1. **Domain mismatch:** ViT was trained on ImageNet objects. Its class token embedding encodes object-level concepts (shape, colour, texture of distinct foreground subjects). Cloud images have no distinct foreground subject — the discriminating signal is diffuse texture, luminance gradients, and macro structure (wispy vs layered vs puffy). These are not the features ImageNet training emphasised.

2. **Fine-grained class similarity:** Several cloud types are visually similar even to expert human eyes (Ac vs Cc, Cs vs As, Sc vs St). A frozen encoder trained on 1000 dissimilar ImageNet objects will not naturally separate these.

The data cleaning addresses the noise problem (dirty labels) but not the representational problem (frozen features). Meaningful further improvement likely requires unfreezing deeper encoder layers, which is compute-bound and better suited to cloud compute (AWS).

---

## Key Design Decisions

- **Cache features, not images:** 768-dim cached embeddings reduce a 300-epoch run from hours to minutes without changing the result.
- **Separate cache directories:** `cached_features/` and `cached_features_filtered/` keep the two runs independent so results can be compared directly.
- **`Ct` threshold set to 0.0:** Contrails only had 200 images in the NASA GLOBE dataset and only 14% passed the 0.9 threshold used elsewhere. Rather than losing this class entirely, all 200 images are included regardless of filter confidence.
- **`ClSk` excluded:** The 11-class task is cloud type identification; clear sky is not a cloud type and is not in the training set.

---

# Checkpoint 4 — Filtered Dataset, Sampling Strategy & Gradual Unfreezing

**Current notebooks:** `08_transfer_learning_filtered.ipynb`, `08_1_transfer_learning_filtered.ipynb`, `08_2_transfer_learning_filtered.ipynb`, `colab_09_transfer_learning_filtered.ipynb`, `colab_10_transfer_learning_poc_unfreeze.ipynb`, `colab_11_transfer_learning_gradual_unfreeze.ipynb`

This checkpoint covers: standardising the filter CSV, building a clean filtered image extract, comparing sampling strategies for cached-embedding training, establishing why end-to-end ViT training must run on Colab, and designing a gradual-unfreezing curriculum for the Colab notebooks.

---

## 1. CSV Column Standardisation

The all-classes filter CSV (`cloud_filter_results.csv`) was saved with inconsistent column names across different runs. The canonical naming convention going forward is:

| Column | Description |
|---|---|
| `folder` | Cloud type label (`Ac`, `Cu`, etc.) |
| `filename` | Image filename |
| `head2_pred` | Binary prediction: `cloud` or `clear_sky` |
| `head2_conf` | head2 softmax probability for `cloud` (0 = clear_sky, 1 = cloud) |
| `head1_pred` | Top-1 ImageNet class from the frozen ResNet18 head1 |
| `head1_conf` | head1 top-1 confidence |

The first clean export under this convention is saved as `cloud_filter_results_1.csv`. All downstream notebooks reference this file. A rename shim (`df.rename(columns={"cloud_conf": "head2_conf", ...})`) is applied at load time in any notebook that may encounter the old column names.

---

## 2. Per-Class Visual Analysis (`resnet18_cloud_filter_analysis.ipynb`)

A standalone analysis notebook that loads `cloud_filter_results_1.csv` and generates four visualisations per cloud class:

1. **head2 confidence distribution** — histogram with threshold and 0.5 lower-bound markers
2. **Not-cloud samples** — 4×4 grid sorted by lowest head2 confidence
3. **Borderline samples (0.5–threshold)** — random sample from the uncertain band
4. **High-confidence cloud samples** — 4×4 grid sorted by highest head2 confidence

Both `head2_pred/head2_conf` and `head1_pred/head1_conf` are shown in each image subtitle, allowing side-by-side comparison of the binary filter and the ImageNet backbone predictions.

Per-class thresholds used throughout:

| Class | Threshold | Notes |
|---|---|---|
| Ac, As, Cb, Cc, Ci, Cs, Cu, Ns, Sc, St | 0.8 | Conservative cutoff |
| Ct | 0.0 | Only 200 images total — include all |

---

## 3. Filtered Image Extract (`_new_dataset_minifier.ipynb`)

Rather than selecting images by CSV lookup at training time, a physical copy of the filtered dataset is built once and reused:

- Reads `cloud_filter_results_1.csv`, applies per-class thresholds
- For each class, **randomly samples** up to 3,000 images from the full `[threshold, 1.0]` confidence range (`random_state=42`)
- Copies selected images to `resources/cloud-images/NASA_GLOBE_CD/extract_filtered/` preserving the per-class folder structure
- Raises `FileExistsError` if the destination already exists (prevents accidental overwrites)

**Why random sample across the full confidence range rather than top-N?** Taking only the highest-confidence images (alphabetical `.head()` or sorted) creates a biased subset skewed toward easy, visually obvious examples. Sampling across the full 0.8–1.0 range includes harder boundary cases that may help generalisation.

**Resulting dataset:** 10 × 3,000 + 200 (Ct) = **30,200 images**

---

## 4. Cached Embedding Variants (Local — Notebooks 08 / 08_1 / 08_2)

Three variants of the cached-embedding training notebook were produced to isolate the effect of image selection strategy vs training noise:

| Notebook | Image selection | `random_state` | Cache dir | Checkpoint |
|---|---|---|---|---|
| `08_transfer_learning_filtered.ipynb` | random sample | 42 | `cached_features_filtered_v2/` | `best_model_filtered.pt` |
| `08_1_transfer_learning_filtered.ipynb` | alphabetical (`.head()`) | 42 | `cached_features_filtered/` | `best_model_filtered_alphabetical.pt` |
| `08_2_transfer_learning_filtered.ipynb` | random sample | 0 | `cached_features_filtered_v3/` | `best_model_filtered_seed0.pt` |

**Key finding:** Confidence distribution analysis showed means within 0.005 of each other across all classes between alphabetical and random selection. The ~3% accuracy difference observed between notebook 08 and 08_1 is attributable to training noise rather than a real effect of sampling strategy. Notebook 08_2 (seed=0) was created to test this hypothesis with a different split seed.

**Cache invalidation:** Each notebook checks for the existence of `{FEATURES_PATH}/train.pt`. If the directory is wiped, the extraction cell re-runs automatically. A utility cell in 08_2 (`shutil.rmtree(FEATURES_PATH)`) allows on-demand cache clearing.

**Note on `num_workers`:** The `T.Lambda` in the augmentation pipeline cannot be pickled by Python's multiprocessing spawn context (macOS default). All DataLoaders use `num_workers=0` locally.

---

## 5. Why End-to-End ViT Training Must Run on Colab

Notebooks `09_transfer_learning_e2e_filtered.ipynb` and `09_1_transfer_learning_e2e_filtered.ipynb` attempt full end-to-end ViT-B-16 fine-tuning locally. A single epoch (21k images × 384×384, batch=32, `num_workers=0`) took over 86 minutes on Apple Silicon MPS without completing — making 300-epoch runs infeasible locally.

Root causes:
- **`num_workers=0`** — all JPEG decoding, resizing (384×384), and augmentation runs in the main process, starving the GPU between batches
- **Full 86M-parameter backprop on MPS** — MPS attention op throughput is significantly lower than CUDA; each backward pass takes several seconds

The correct local approach remains the cached-embedding route (notebooks 08/08_1/08_2): extract once (~30 min), then each epoch is 2–5 seconds. End-to-end training is reserved for Colab GPU sessions.

---

## 6. Colab Notebook Family

All Colab notebooks are based on `colab_06_1_transfer_learning.ipynb` (uses `IMAGENET1K_SWAG_LINEAR_V1` weights, `CUDA_LAUNCH_BLOCKING=1`, tqdm progress bars inside the train loop).

| Notebook | Description |
|---|---|
| `colab_09_transfer_learning_filtered.ipynb` | Direct port of `colab_06_1` to `extract_filtered/` — heads-only training, 30 × 10 epochs |
| `colab_10_transfer_learning_poc_unfreeze.ipynb` | Proof-of-concept gradual unfreezing — 10/5/5 epochs, no early stopping |
| `colab_11_transfer_learning_gradual_unfreeze.ipynb` | Full gradual unfreezing — 100/50/30 max epochs with early stopping per phase |

The only changes from `colab_06_1` in `colab_09`:

| Field | `colab_06_1` | `colab_09` |
|---|---|---|
| `unzipped_images_dir` | `extract` | `extract_filtered` |
| `zip_path` | `extract.zip` | `extract_filtered.zip` |
| `expected_image_count` | 30559 | 30200 |
| `IMAGES_PATH` | `extract` | `extract_filtered` |
| `n_classes` / `num_classes` | 12 | 11 |

---

## 7. Gradual Unfreezing Strategy (Colab 10 & 11)

ViT-B-16 has 12 transformer encoder layers (`model.encoder.layers[0]–[11]`). The standard approach is to unfreeze from the top down — the last layers are most task-specific and adapt fastest. Three phases:

**Phase 1 — heads only**
```python
optimizer = torch.optim.AdamW(model.heads.parameters(), lr=1e-3)
```

**Phase 2 — unfreeze last encoder block (layer 11)**
```python
for param in model.encoder.layers[-1].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": model.heads.parameters(),              "lr": 1e-4},
    {"params": model.encoder.layers[-1].parameters(), "lr": 1e-5},
])
```

**Phase 3 — unfreeze last 4 encoder blocks (layers 8–11)**
```python
for layer in model.encoder.layers[-4:-1]:  # layers 8, 9, 10
    for param in layer.parameters():
        param.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": model.heads.parameters(),                                           "lr": 1e-4},
    {"params": model.encoder.layers[-1].parameters(),                              "lr": 1e-5},
    {"params": [p for l in model.encoder.layers[-4:-1] for p in l.parameters()],  "lr": 5e-6},
])
```

**LR rationale:** Each newly unfrozen group gets a 10× smaller LR than the previous phase. Pretrained weights are valuable; small gradients adjust them without destroying the representations learned during ImageNet training. Going deeper than the last 4 blocks is rarely worth it for a classification task of this scale.

| Notebook | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| `colab_10` (PoC) | 10 epochs | 5 epochs | 5 epochs |
| `colab_11` (full) | 100 epochs, patience 20 | 50 epochs, patience 15 | 30 epochs, patience 10 |

`colab_11` saves a checkpoint at each phase (`best_model_phase{1,2,3}.pt`) and restores the best weights before moving to the next phase. The plot cell at the end renders all three phases on a single continuous axis with colour-coded train/val curves.

---

## Key Design Decisions

- **Random sampling across full confidence range:** Avoids selection bias toward the easiest images, at the cost of including harder borderline examples that test generalisation.
- **Physical copy (`extract_filtered/`) rather than CSV lookup at training time:** Simplifies the Colab data pipeline — zip the directory, upload to Drive, extract. No CSV dependency at runtime.
- **`colab_06_1` as the Colab base (not `colab_06`):** `colab_06_1` uses `IMAGENET1K_SWAG_LINEAR_V1` (224×224 linear probe weights) rather than `E2E_V1` (384×384 end-to-end weights), and includes `CUDA_LAUNCH_BLOCKING=1` for stable CUDA error reporting.
- **Discriminative learning rates across unfreezing phases:** Rather than a single global LR for all unfrozen layers, each layer group gets its own LR (1e-4 / 1e-5 / 5e-6). Earlier layers get smaller LRs to preserve low-level features while allowing higher layers to specialise.
- **`Ct` handled by `WeightedRandomSampler`:** With only 200 Ct images vs 3,000 for other classes, the sampler gives Ct images ~15× the sampling weight. Each Ct image is seen ~15 times per epoch with fresh augmentation, providing variation despite the small pool.

---

## Project Structure (updated)

```
project-root/
├── models/
│   └── multihead_resnet18_cloud_sky.pth
├── resources/
│   └── cloud-images/
│       └── NASA_GLOBE_CD/
│           ├── downloaded_images/          ← original raw images
│           ├── extract_filtered/           ← filtered copy (30,200 images)
│           │   ├── Ac/  As/  Cb/  Cc/  Ci/
│           │   ├── Cs/  Ct/  Cu/  Ns/
│           │   └── Sc/  St/
│           ├── cloud_filter_results.csv    ← original filter output
│           └── cloud_filter_results_1.csv  ← canonical filter CSV (standardised columns)
└── notebooks/
    ├── _new_dataset_minifier.ipynb
    ├── resnet18_cloud_filter_analysis.ipynb
    ├── 08_transfer_learning_filtered.ipynb         ← cached, random sample, seed=42
    ├── 08_1_transfer_learning_filtered.ipynb       ← cached, alphabetical
    ├── 08_2_transfer_learning_filtered.ipynb       ← cached, random sample, seed=0
    ├── 09_transfer_learning_e2e_filtered.ipynb     ← end-to-end local (impractical)
    ├── 09_1_transfer_learning_e2e_filtered.ipynb   ← end-to-end local, dir-scan loader
    ├── colab_09_transfer_learning_filtered.ipynb   ← Colab, heads only
    ├── colab_10_transfer_learning_poc_unfreeze.ipynb   ← Colab, PoC unfreezing
    └── colab_11_transfer_learning_gradual_unfreeze.ipynb ← Colab, full unfreezing
```

---

# Checkpoint 5 — GPU Training, Overfitting Analysis & Regularisation

**Current notebooks:** `colab_11_1`, `colab_11_2`, `colab_11_3`

This checkpoint covers: bug fixes discovered during Colab runs, end-to-end training results on L4 and A100 GPUs, overfitting analysis, and a regularised variant.

---

## 1. Bug Fixes

### WeightedRandomSampler not wired up
`WeightedRandomSampler` was defined in the DataLoader cell but never passed to `DataLoader` — `shuffle=True` was used instead. This meant Ct (only 140 training images) was never oversampled. Fixed in `colab_09`, `colab_10`, `colab_11`, `colab_11_1`, `colab_11_2`, `colab_11_3`:

```python
# before
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# after
train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)
```

### T.Lambda pickling error (Python 3.12 / forkserver)
Python 3.12 changed the default multiprocessing start method on Linux to `forkserver`, which requires all DataLoader worker arguments to be picklable. `T.Lambda(lambda ...)` is not picklable. Replaced with a named class in all Colab notebooks:

```python
class WrapTranslation:
    def __init__(self, max_frac): self.max_frac = max_frac
    def __call__(self, x):
        h, w = x.shape[-2], x.shape[-1]
        shift_h = int(torch.randint(-int(self.max_frac * h), int(self.max_frac * h) + 1, (1,)).item())
        shift_w = int(torch.randint(-int(self.max_frac * w), int(self.max_frac * w) + 1, (1,)).item())
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))
```

Despite this fix, `num_workers > 0` still triggers `AssertionError: can only test a child process` during DataLoader teardown on Colab. All Colab notebooks use `num_workers=0`.

---

## 2. colab_09 Frozen Baseline Results (L4)

4 runs × 10 epochs (40 total epochs), heads-only, `IMAGENET1K_SWAG_LINEAR_V1`:

| Run | Test accuracy |
|---|---|
| 1 | 0.3263 |
| 2 | 0.3265 |
| 3 | 0.3280 |
| 4 | 0.3180 |

**Conclusion:** Frozen encoder ceiling confirmed at ~33%. Consistent across runs — this is a structural limit, not noise. Unfreezing is required to progress.

---

## 3. colab_10 PoC Unfreezing Results (L4, ~20 min/epoch)

| Phase | Best val | Train at stop | Epochs |
|---|---|---|---|
| Phase 1 — heads only | 0.3490 | ~0.41 | 10 |
| Phase 2 — + layers[-1] | 0.3702 | ~0.45 | 5 |
| Phase 3 — + layers[-4:] | ~0.36 | ~0.52 | 5 |
| **Test accuracy** | **0.3707** | | |

Phase 2 produced a clear jump (+4pp over frozen baseline). Phase 3 showed early overfitting. No early stopping or checkpointing in this PoC notebook.

---

## 4. colab_11_2 Full Run Results (A100, bfloat16, batch_size=128, ~5 min/epoch)

| Phase | Best val | Train at stop | Gap | Epochs to stop |
|---|---|---|---|---|
| Phase 1 — heads only | 0.3490 | ~0.43 | ~8% | 30 |
| Phase 2 — + layers[-1] | 0.3755 | ~0.60 | ~22% | 40 |
| Phase 3 — + layers[-4:] | **0.3755** | ~0.68 | ~30% | 15 |
| **Test accuracy** | **0.3626** | | | |

**Key finding: Phase 3 gave zero val improvement over Phase 2.** Unfreezing 4 blocks only added overfitting (30% train/val gap) without moving the ceiling. The model memorised training data but could not generalise further.

---

## 5. Overfitting Analysis

The 30% train/val gap in Phase 3 indicates the model has more capacity than the data can support. Root causes:

- **Dataset size:** 30,200 images (~21,000 train) is insufficient for fine-grained 11-class cloud classification with a 28M-parameter model
- **Inter-class ambiguity:** Several cloud types (Ac/Cc, Cs/As, Sc/St) are visually similar even to experts; the model cannot reliably discriminate them with the available training signal
- **Ct constraint:** Only 200 total Ct images (140 train). Despite WeightedRandomSampler's 15× oversampling, the variety is fundamentally limited

**Data scaling analysis:** The current dataset uses 3,000 images/class at a 0.99 confidence threshold. Pool sizes at this threshold:

| Class | Pool at 0.99 |
|---|---|
| Cb | 4,617 ← bottleneck |
| Cc | 5,500 |
| Cs | 7,039 |
| Ci | 8,692 |
| others | 8,354–58,460 |
| Ct | 200 ← hard ceiling |

Lowering the threshold to 0.95 would unlock significantly more images for the bottleneck classes. Target: 5,000–10,000 images/class, which is the typical range where ViT fine-tuning generalises well.

---

## 6. Notebook Variants

| Notebook | GPU | Key differences from colab_11 |
|---|---|---|
| `colab_11_transfer_learning_gradual_unfreeze.ipynb` | L4 | Baseline full run |
| `colab_11_1_transfer_learning_gradual_unfreeze_shm.ipynb` | L4 | Images extracted to `/dev/shm` (RAM-backed) |
| `colab_11_2_transfer_learning_gradual_unfreeze_a100.ipynb` | A100 | bfloat16 AMP, batch_size=128 |
| `colab_11_3_transfer_learning_gradual_unfreeze_regularised.ipynb` | A100 | + label_smoothing=0.1, weight_decay=0.05 |

### /dev/shm (colab_11_1)
Extracts the zip to `/dev/shm` (RAM filesystem) instead of `/content` (SSD). With `num_workers=0` the benefit is minimal — CPU transforms dominate loading time, not disk reads. Would matter more if workers were available.

### bfloat16 AMP (colab_11_2)
A100 Tensor Cores are optimised for bfloat16. Wrapping forward pass and loss in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` gives significant speedup with no GradScaler needed (bfloat16 has better numerical range than float16). Epoch time: ~5 min vs ~20 min on L4.

### Regularisation (colab_11_3)
Two changes targeting the overfitting:
- `nn.CrossEntropyLoss(label_smoothing=0.1)` — softens targets, penalises overconfident predictions
- `weight_decay=0.05` on all AdamW optimisers (up from default 0.01)

Results pending.

---

## Key Design Decisions

- **Phase 3 may not be worth running:** colab_11_2 showed Phase 3 adds no val improvement over Phase 2 while dramatically increasing overfitting. Future runs should consider stopping at Phase 2 and investing compute in more data instead.
- **More data is a higher-priority fix than regularisation:** The train/val gap is a data problem. Regularisation (colab_11_3) is worth trying but unlikely to break through the ~0.38 ceiling without more training examples.
- **A100 > L4 for this workload:** 4× speedup per epoch (5 min vs 20 min) makes the full 100/50/30 epoch budget practical. bfloat16 is free accuracy-wise.

---

## Project Structure (updated)

```
notebooks/
├── colab_09_transfer_learning_filtered.ipynb
├── colab_10_transfer_learning_poc_unfreeze.ipynb
├── colab_11_transfer_learning_gradual_unfreeze.ipynb
├── colab_11_1_transfer_learning_gradual_unfreeze_shm.ipynb       ← /dev/shm extraction
├── colab_11_2_transfer_learning_gradual_unfreeze_a100.ipynb      ← A100, bfloat16
└── colab_11_3_transfer_learning_gradual_unfreeze_regularised.ipynb ← label smooth + wd
```

---

# Checkpoint 6 — CNN Backbone Selection (Replicating Paper Figure 7)

**Current notebook:** `13_transfer_learning_densenet121_swimcat.ipynb`

This checkpoint replicates the backbone selection experiment from *CloudDenseNet: Lightweight Ground-Based Cloud Classification* (Li et al., Sensors 2023). Before designing their CloudDenseNet architecture, the paper benchmarked 9 CNN backbones under identical conditions to justify their choice of DenseNet121.

---

## Experiment Setup

The paper trained all 9 models on SWIMCAT — a small, clean dataset of 784 ground-based sky images across 5 broad categories (clear sky, patterned clouds, thick dark clouds, thick white clouds, veil clouds). It was chosen specifically because it's lightweight enough to benchmark multiple architectures quickly.

All models used identical settings:

- ImageNet pre-trained weights
- Backbone fully frozen — only the classification head trained
- Same hyperparameters across all models

The paper used a 50/50 train/test split with no validation set. This notebook uses a 70/20/10 split to retain a validation set for early stopping, which is a strictly better setup — the results aren't directly comparable to the paper's 95.51% but are close enough to validate the implementation.

**Note:** This notebook uses **SWIMCAT-extend** (not the original SWIMCAT). SWIMCAT-extend is an expanded version of the dataset with 2,100 images across 6 categories (the original 5 plus F-Veil Clouds), with 350 images per class. The dataset is located at `resources/cloud-images/Swimcat-extend/`.

---

## Results

| Metric | Value |
|---|---|
| Best val accuracy | **0.9810** (epoch 47) |
| Test accuracy | **0.9762** |
| Early stop epoch | 67 (patience 20) |

Training converged fast: 93.3% val by epoch 3, 95%+ by epoch 7. The model essentially plateaued in the 97–98% range from epoch 47 onwards with no further improvement.

Train/val gap was negligible (~97.5% train vs 98.1% val at peak) — no overfitting. The frozen backbone + linear head is well regularised for a small clean dataset of this size.

**Comparison to paper:** The paper reports 95.51% for DenseNet121 on the original SWIMCAT with a 50/50 split. Our 97.62% test accuracy on SWIMCAT-extend with a stricter 70/20/10 split is consistent with and exceeds that result, validating the implementation. The higher accuracy is likely a combination of the larger dataset (2,100 vs 784 images) and the cleaner 6-class taxonomy of SWIMCAT-extend.

---

## Key Design Decisions

- **Plain shuffle over `WeightedRandomSampler`:** SWIMCAT-extend is perfectly balanced (350/class), so weighted sampling adds no benefit.
- **`model.eval()` during training:** DenseNet121's frozen BatchNorm layers must stay in eval mode to keep running stats fixed. The classifier head is a plain Linear so train/eval mode makes no difference for it.
- **6 classes instead of 5:** SWIMCAT-extend adds F-Veil Clouds vs the original SWIMCAT. The paper's 95.51% benchmark used 5 classes; direct comparison is approximate.

---

## Project Structure (updated)

```
notebooks/
└── 13_transfer_learning_densenet121_swimcat.ipynb   ← DenseNet121, SWIMCAT-extend, head only
```

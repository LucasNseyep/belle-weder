import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


class AugmentingClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps any sklearn classifier so that augmentation happens *inside* fit() only.
    Validation folds always see unaugmented originals — no data leakage.

    Parameters
    ----------
    base_clf : sklearn estimator
    augmentation_transform : callable
        Applied to each image tensor during oversampling. Must accept a
        (1, H, W) torch.Tensor and return a (1, H, W) torch.Tensor.
    target_count : int
        Target number of samples per class after augmentation.
    random_state : int
    """

    def __init__(self, base_clf=None, augmentation_transform: Optional[Callable] = None,
                 target_count: int = 2400, random_state: int = 42):
        self.base_clf = base_clf
        self.augmentation_transform = augmentation_transform
        self.target_count = target_count
        self.random_state = random_state

    def _augment(self, X, y):
        import torch
        X_extra, y_extra = [], []

        if self.augmentation_transform is not None:
            for cls in np.unique(y):
                cls_imgs = X[y == cls]
                multiple = self.target_count // len(cls_imgs)
                for img in cls_imgs:
                    img_torch = torch.from_numpy(img)
                    for _ in range(multiple - 1):
                        aug = self.augmentation_transform(
                            img_torch.unsqueeze(0)
                        ).squeeze(0).numpy()
                        X_extra.append(aug)
                        y_extra.append(cls)

        X_norm = X.astype(np.float32) / 255.0
        if X_extra:
            X_all = np.concatenate([X_norm, np.stack(X_extra)])
            y_all = np.concatenate([y, np.array(y_extra)])
        else:
            X_all, y_all = X_norm, y

        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(len(X_all))
        return X_all[idx], y_all[idx]

    def fit(self, X, y):
        X_aug, y_aug = self._augment(X, y)
        self.base_clf.fit(X_aug.reshape(len(X_aug), -1), y_aug)
        return self

    def predict(self, X):
        X_norm = X.astype(np.float32) / 255.0
        return self.base_clf.predict(X_norm.reshape(len(X_norm), -1))

    def predict_proba(self, X):
        X_norm = X.astype(np.float32) / 255.0
        return self.base_clf.predict_proba(X_norm.reshape(len(X_norm), -1))


def _get_default_classifiers(random_state: int = 42) -> dict:
    return {
        "Gaussian NB":          GaussianNB(),
        "Logistic Regression":  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)),
        "Linear SVC (SGD)":     make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", random_state=random_state)),
        "SVC (RBF)":            make_pipeline(StandardScaler(), SVC(probability=True, random_state=random_state)),
        "KNN":                  make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Decision Tree":        DecisionTreeClassifier(random_state=random_state),
        "Random Forest":        RandomForestClassifier(random_state=random_state),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.7, random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
        "MLP":                  make_pipeline(StandardScaler(), MLPClassifier(max_iter=500, random_state=random_state)),
    }


def evaluate_classifiers(
    X, y,
    augmentation_transform: Optional[Callable] = None,
    scoring: str = "f1_macro",
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Evaluate a standard suite of sklearn classifiers with stratified cross-validation.

    Augmentation (if provided) is applied inside each CV fold via AugmentingClassifier
    so validation folds always see unaugmented originals.

    Parameters
    ----------
    X : array of shape (n, H, W)
        Raw (non-augmented, non-normalised) images.
    y : array of shape (n,)
        Integer class labels.
    augmentation_transform : callable, optional
        Passed to AugmentingClassifier. None disables augmentation.
    scoring : str
        sklearn scoring string. Default "f1_macro" is appropriate for imbalanced classes.
    cv : int
        Number of CV folds.
    n_jobs : int
        Parallelism for cross_val_score. Set to 1 when using MPS/CUDA augmentation.
    random_state : int
    verbose : int

    Returns
    -------
    pd.DataFrame sorted by mean_score descending.
    """
    classifiers = _get_default_classifiers(random_state)
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []

    for name, clf in tqdm(classifiers.items(), total=len(classifiers)):
        if verbose:
            print(f"\nEvaluating: {name}")
        t0 = time.perf_counter()
        scores = cross_val_score(
            AugmentingClassifier(base_clf=clf, augmentation_transform=augmentation_transform),
            X, y,
            scoring=scoring,
            cv=cv_obj,
            n_jobs=n_jobs,
        )
        results.append({
            "model":        name,
            "mean_score":   np.mean(scores),
            "std_score":    np.std(scores),
            "time_elapsed": time.perf_counter() - t0,
        })

    return pd.DataFrame(results).sort_values("mean_score", ascending=False)

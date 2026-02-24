"""
PoseDataset — species-agnostic replacement for ``ZebData``.

``ZebData`` is retained as an alias for backwards compatibility.
All augmentation methods delegate to ``poser.core.augmentation`` so
there is no duplicated code.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.utils import class_weight as sk_class_weight

from .augmentation import (
    rotate_transform,
    jitter_transform,
    scale_transform,
    shear_transform,
    roll_transform,
    fragment_transform,
    random_augmentation,
    dynamic_augmentation,
    _pad_to_length,
)


class PoseDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for pose sequences.

    Data shape convention: ``(N, C, T, V, M)``
    — samples × channels (x/y/ci) × time × vertices × individuals.

    Parameters
    ----------
    data_file:
        Path to a ``.npy`` file with shape ``(N, C, T, V, M)`` or
        ``(C, T, V, M)`` (single continuous recording).
    label_file:
        Path to a ``.npy`` file of integer or float labels, shape ``(N,)``.
    transform:
        String or list of strings. Supported tokens:
        ``"center"``, ``"align"``, ``"pad"``, ``"pad_flip"``, ``"flipy"``,
        ``"heatmap"``.
    augment:
        Whether to apply random augmentation in ``__getitem__``.
    ideal_sample_no:
        If provided, ``dynamic_augmentation()`` oversamples to this count.
    center_node, head_node:
        Node indices used by centering / alignment transforms.
    T:
        Target padded time-dimension length.
    binary, binary_class:
        Binary classification mode.
    regress:
        Regression mode — labels loaded as float64.
    labels_to_ignore:
        Class labels to drop.
    label_dict:
        ``{original_label: new_label}`` remapping.
    preprocess_frame:
        If True, ``data_file`` is a 4D ``(C, T, V, M)`` continuous recording
        and ``__getitem__`` windows around the requested frame index.
    window_size:
        Half-window size when ``preprocess_frame=True``.
    """

    def __init__(
        self,
        data_file: Optional[str] = None,
        label_file: Optional[str] = None,
        transform=None,
        target_transform=None,
        ideal_sample_no: Optional[int] = None,
        augment: bool = False,
        shift: bool = False,
        labels_to_ignore=None,
        label_dict=None,
        regress: bool = False,
        center_node: int = 0,
        head_node: int = 0,
        T: Optional[int] = None,
        lazy: bool = True,
        binary: bool = False,
        binary_class=None,
        preprocess_frame: bool = False,
        window_size: Optional[int] = None,
    ):
        self.ideal_sample_no = ideal_sample_no
        self.transform = transform
        self.target_transform = target_transform
        self.center_node = center_node
        self.head_node = head_node
        self.T = T
        self.augment = augment
        self.preprocess_frame = preprocess_frame
        self.window_size = window_size
        print(f"PoseDataset — augment={self.augment}")

        if data_file is not None:
            filesize = os.path.getsize(data_file) / 1e9
            available = psutil.virtual_memory().available / 1e9
            mmap = filesize > available
            if mmap:
                print("File larger than available RAM — using memory-mapped mode.")
            self.data = np.load(data_file, mmap_mode="r" if mmap else None)

            if self.data.shape[0] == 1:
                self.data = self.data.reshape(*self.data.shape[1:])

            if self.data.ndim == 4:
                self.C, self.T0, self.V, self.M = self.data.shape
                print(f"4D data: C={self.C} T={self.T0} V={self.V} M={self.M}")
            elif self.data.ndim == 5:
                self.N, self.C, self.T0, self.V, self.M = self.data.shape
                print(f"5D data: N={self.N} C={self.C} T={self.T0} V={self.V} M={self.M}")

            # --- Labels ---
            if regress:
                self.labels = np.load(label_file).astype("float64")
            elif binary:
                self.labels = np.load(label_file).astype("float64")
                if self.labels.shape[0] == 1:
                    self.labels = self.labels.reshape(*self.labels.shape[1:])
                if binary_class is not None:
                    mask = np.isin(self.labels, binary_class)
                    self.labels = mask.astype("float64")
            else:
                self.labels = np.load(label_file).astype("int64")
                if self.labels.shape[0] == 1:
                    self.labels = self.labels.reshape(*self.labels.shape[1:])
                print(f"Label breakdown:\n{pd.Series(self.labels).value_counts()}")

                if labels_to_ignore is not None:
                    keep = ~np.isin(self.labels, labels_to_ignore)
                    self.labels = self.labels[keep]
                    self.data = self.data[keep]

                if label_dict is None:
                    mapping = {k: v for v, k in enumerate(np.unique(self.labels))}
                else:
                    mapping = label_dict
                new_labels = np.zeros_like(self.labels)
                for orig, new in mapping.items():
                    new_labels[self.labels == orig] = new
                self.labels = new_labels
                print(f"Label mapping: {mapping}")
                print(f"Unique labels: {np.unique(self.labels)}")

            if shift:
                self.data = self.data + np.array([50, 50])

        else:
            self.data = None
            self.labels = None

        # Heatmap grid (CuPy-accelerated, optional)
        if self.transform == "heatmap":
            self._init_heatmap_grid()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if self.preprocess_frame:
            behaviour = self.preprocess_frames(idx, self.window_size)
        else:
            behaviour = self.data[idx].copy()
        label = self.labels[idx]

        if self.transform is not None:
            if "flipy" in self.transform:
                behaviour = self.flipy(behaviour)
            if "center" in self.transform:
                behaviour = self.center_all(behaviour, self.center_node)
            if "align" in self.transform:
                behaviour = self.align(behaviour)
            if "pad" in self.transform and "pad_flip" not in self.transform:
                behaviour = self.pad(behaviour, self.T)
            if "pad_flip" in self.transform:
                behaviour = self.pad_flip(behaviour, self.T)
            if self.augment:
                behaviour = random_augmentation(behaviour)
            if self.transform == "heatmap":
                behaviour = self._to_heatmap(behaviour)

        if not isinstance(behaviour, torch.Tensor):
            behaviour = torch.from_numpy(behaviour).to(torch.float32)

        if label.dtype != "float64" if hasattr(label, "dtype") else False:
            label = torch.tensor(label).long()
        else:
            label = torch.tensor(label)
        return behaviour, label

    # ------------------------------------------------------------------
    # Frame-level preprocessing (continuous 4D data)
    # ------------------------------------------------------------------

    def preprocess_frames(self, frame: int, window_size: int) -> np.ndarray:
        hw = window_size // 2
        t0, t1 = frame - hw, frame + hw
        left_pad = right_pad = None
        if t0 < 0:
            left_pad = np.zeros((self.C, abs(t0), self.V, self.M))
            t0 = 0
        if t1 > self.T0:
            right_pad = np.zeros((self.C, t1 - self.T0, self.V, self.M))
            t1 = self.T0
        bhv = self.data[:, t0:t1].copy()
        if left_pad is not None:
            bhv = np.concatenate([left_pad, bhv], axis=1)
        if right_pad is not None:
            bhv = np.concatenate([bhv, right_pad], axis=1)
        return bhv

    # ------------------------------------------------------------------
    # Pose transforms (instance methods — delegate to core functions)
    # ------------------------------------------------------------------

    def flipy(self, bhv: np.ndarray) -> np.ndarray:
        result = bhv.copy()
        result[1] *= -1
        return result

    def align(self, bhv: np.ndarray) -> np.ndarray:
        from .preprocessing import align as _align
        return _align(bhv, self.head_node)

    def center_all(self, bout: np.ndarray, center_node: int) -> np.ndarray:
        from .preprocessing import center_all as _center
        return _center(bout, center_node)

    def center_first(self, bout: np.ndarray, center_node: int) -> np.ndarray:
        ref = bout[0:2, 0, center_node].reshape(2, 1, 1, -1)
        result = bout.copy()
        result[0:2] -= ref
        return result

    def pad(self, bout: np.ndarray, new_t: int) -> np.ndarray:
        return _pad_to_length(bout, new_t)

    def pad_flip(self, bout: np.ndarray, new_t: int) -> np.ndarray:
        from .preprocessing import pad_flip_bout
        return pad_flip_bout(bout, new_t)

    # Augmentation methods — delegate to module-level functions
    def rotate_transform(self, b, n): return rotate_transform(b, n)
    def jitter_transform(self, b, n): return jitter_transform(b, n)
    def scale_transform(self, b, n): return scale_transform(b, n)
    def shear_transform(self, b, n): return shear_transform(b, n)
    def roll_transform(self, b, n): return roll_transform(b, n)
    def fragment_transform(self, b, n): return fragment_transform(b, n)

    def random_augmentation(self, bhv: np.ndarray, num_aug: int = 1) -> np.ndarray:
        return random_augmentation(bhv, num_aug)

    def dynamic_augmentation(self) -> None:
        if self.ideal_sample_no is None:
            raise ValueError("ideal_sample_no must be set before calling dynamic_augmentation()")
        self.data, self.labels = dynamic_augmentation(
            self.data, self.labels, self.ideal_sample_no
        )

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        cw = sk_class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(np.sort(self.labels)),
            y=self.labels,
        )
        return torch.tensor(cw, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        cw = self.get_class_weights()
        mapping = {k: v.item() for k, v in enumerate(cw)}
        weights = pd.Series(self.labels).map(mapping).to_numpy(dtype="float32")
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Heatmap (CuPy-accelerated)
    # ------------------------------------------------------------------

    def _init_heatmap_grid(self):
        try:
            import cupy as cp
            x_min, x_max, y_min, y_max = -3, 3, -3, 3
            self.H = self.W = 64
            x = cp.linspace(x_min, x_max, num=self.W, dtype="float32")
            y = cp.linspace(y_min, y_max, num=self.H, dtype="float32")
            xv, yv = cp.meshgrid(x, y, indexing="xy")
            self.xv_rs = xv.reshape((*xv.shape, 1))
            self.yv_rs = yv.reshape((*yv.shape, 1))
        except ImportError:
            print("CuPy not available — heatmap transform disabled.")
            self.xv_rs = self.yv_rs = None

    def _to_heatmap(self, bout):
        try:
            import cupy as cp
            bout_cp = cp.asarray(bout[:, ::4])
            C, T, V, M = bout_cp.shape
            zx = (self.xv_rs - bout_cp[0, :, :, 0].ravel()) ** 2
            zy = (self.yv_rs - bout_cp[1, :, :, 0].ravel()) ** 2
            zz = cp.exp(-(zx + zy) / (2 * 0.01)) * bout_cp[2, :, :, 0].ravel()
            zz = cp.swapaxes(cp.swapaxes(zz.reshape((self.W, self.H, T, V)), 0, -1), 1, 2)
            return torch.as_tensor(zz, device="cpu", dtype=torch.float32)
        except Exception:
            return bout


# ---------------------------------------------------------------------------
# Backwards compat alias
# ---------------------------------------------------------------------------

ZebData = PoseDataset

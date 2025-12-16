import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Union, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import cdist
import ot


class PerturbPairData(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        obs: pd.DataFrame,
        perturb_col: str,
        control_col: str,
        condition_col: str,
        perturb_to_idx: dict,
        condition_to_idx: dict,
        seed: int = 42,
        device=None,
    ):
        self.device = device or torch.device("cpu")
        self.X = X.to(self.device)
        X_cpu = self.X.cpu().numpy()
        self.obs = obs.reset_index(drop=True)

        if condition_col:
            self.conditions = sorted(self.obs[condition_col].unique())
            if not condition_to_idx:
                self.condition_to_idx = {c: i for i, c in enumerate(self.conditions)}
            else:
                self.condition_to_idx = condition_to_idx
        else:
            self.conditions = None

        self.perturbs = sorted(self.obs[perturb_col].unique())
        if not perturb_to_idx:
            self.perturb_to_idx = {p: i for i, p in enumerate(self.perturbs)}
        else:
            self.perturb_to_idx = perturb_to_idx

        self.ctrl_idx = {}
        self.pert_idx = {}

        self.valid_keys = []

        self.pairs = []

        for perturbation in self.perturbs:
            perturb_rows = self.obs[perturb_col] == perturbation
            if self.conditions:
                for condition in self.conditions:
                    condition_rows = self.obs[condition_col] == condition
                    control_rows = self.obs[control_col].astype(bool) & condition_rows
                    control_mask = control_rows
                    ctrl = control_mask
                    pert = perturb_rows & condition_rows & (~control_mask)
                    ctrl_idx = np.where(ctrl)[0]
                    pert_idx = np.where(pert)[0]

                    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
                        continue

                    # compute optimal transport pairing
                    M = cdist(X_cpu[ctrl_idx], X_cpu[pert_idx], metric="sqeuclidean")

                    a = np.ones((len(ctrl_idx),)) / len(ctrl_idx)
                    b = np.ones((len(pert_idx),)) / len(pert_idx)
                    gamma = ot.emd(a, b, M)
                    n_samples = max(len(ctrl_idx), len(pert_idx))
                    flat_gamma = gamma.flatten()
                    flat_gamma = flat_gamma / flat_gamma.sum()

                    rng = np.random.default_rng(seed)
                    indices_pair = rng.choice(len(flat_gamma), size=n_samples, p=flat_gamma, replace=True)

                    rows = indices_pair // len(pert_idx)
                    cols = indices_pair % len(pert_idx)

                    final_control = ctrl_idx[rows]
                    final_perturb = pert_idx[cols]

                    for i in range(len(final_control)):
                        self.pairs.append((final_control[i], final_perturb[i], perturbation, condition))

            else:
                raise NotImplementedError

        print(f"generated {len(self.pairs)} ot paris")

    def __len__(self):
        return len(self.pairs)

    def num_conditions(self):
        return len(self.condition_to_idx.keys())

    def __getitem__(self, idx):
        c, q, p, cond = self.pairs[idx]
        return {
            "x_0": self.X[c],
            "x_1": self.X[q],
            "perturb": self.perturb_to_idx[p],
            "condition": self.condition_to_idx[cond],
        }


import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


class PhateDataset(Dataset):
    """
    Custom Phate dataset based on PSET 2 starter code from CPSC6440
    """

    def __init__(self, X, X_phate, raw, device=None):
        self.device = device or torch.device("cpu")

        self.raw = torch.from_numpy(raw.astype(np.float32)).to(self.device)
        self.X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.X_phate = torch.from_numpy(X_phate.astype(np.float32)).to(self.device)

        phate_distances_np = pairwise_distances(X_phate.astype(np.float32), metric="euclidean").astype(np.float32)
        self.phate_distances = torch.from_numpy(phate_distances_np).to(self.device)

        self.n_samples = self.X.shape[0]

        print(f"Dataset initialized with {self.n_samples} samples")
        print(f"PCA data shape: {self.X.shape}")
        print(f"PHATE embedding shape: {self.X_phate.shape}")
        print(f"PHATE distance matrix shape: {self.phate_distances.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {"raw": self.raw[idx], "x": self.X[idx], "phate": self.X_phate[idx], "index": idx}

    def mask_vector(self, indices):
        # batch_types = self.cell_types[indices]
        # mask_matrix = batch_types.unsqueeze(1) == batch_types.unsqueeze(0)
        n = len(indices)
        rows, cols = torch.triu_indices(n, n, offset=1, device=self.device)
        return torch.ones(rows.shape[0], dtype=torch.bool, device=self.device)
        # return mask_matrix[rows, cols]

    def distance_vector(self, indices):
        batch_distances = self.phate_distances[indices][:, indices]
        n = len(indices)
        rows, cols = torch.triu_indices(n, n, offset=1, device=self.device)
        return batch_distances[rows, cols]


class GeodesicDataset(Dataset):
    def __init__(self, X, raw, distance_matrix, device=None):
        self.device = device or torch.device("cpu")
        self.X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.raw = torch.from_numpy(raw.astype(np.float32)).to(self.device)
        # store full NxN distance matrix
        self.distance_matrix = torch.from_numpy(distance_matrix.astype(np.float32)).to(self.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "raw": self.raw[idx], "index": idx}

    def distance_vector(self, indices):
        batch_distances = self.distance_matrix[indices][:, indices]
        n = len(indices)
        rows, cols = torch.triu_indices(n, n, offset=1, device=self.device)
        return batch_distances[rows, cols]

    def mask_vector(self, indices):
        n = len(indices)
        rows, cols = torch.triu_indices(n, n, offset=1, device=self.device)
        return torch.ones(rows.shape[0], dtype=torch.bool, device=self.device)

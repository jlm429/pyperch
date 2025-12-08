import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def make_loaders(
    X,
    y,
    *,
    batch_size: int = 64,
    valid_batch_size: int | None = None,
    valid_split: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
    stratify: bool = False,
    normalize: str | None = None,  # None | "standard" | "minmax"
    return_scaler: bool = False,  # If True, also return the scaler object
):
    """
    Create train and validation DataLoaders from numpy or torch tensors.

    Args:
        X, y: numpy arrays or tensors
        batch_size: training batch size
        valid_batch_size: validation batch size (defaults to batch_size)
        valid_split: fraction of data used for validation
        shuffle: shuffle training data
        seed: RNG seed for reproducible splits
        stratify: whether to stratify (requires classification y values)
        normalize: optional normalization:
            - None      (no scaling)
            - "standard" (mean=0, std=1)
            - "minmax"  (scaled to 0–1)
        return_scaler: if True, return (train_loader, valid_loader, scaler)

    Returns:
        train_loader, valid_loader
        OR
        train_loader, valid_loader, scaler (if return_scaler=True)
    """

    # Convert numpy -> torch
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # ------------------------------------------------------
    # STRATIFIED OR STANDARD SPLIT
    # ------------------------------------------------------
    if stratify:
        if y.dim() != 1:
            raise ValueError("Stratified split requires 1D target array.")

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=valid_split, random_state=seed
        )
        idx_train, idx_valid = next(splitter.split(X, y))
        X_train, X_valid = X[idx_train], X[idx_valid]
        y_train, y_valid = y[idx_train], y[idx_valid]

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=valid_split, random_state=seed
        )

    # ------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------
    scaler = None

    if normalize is not None:
        if normalize == "standard":
            scaler = StandardScaler()
        elif normalize == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("normalize must be None, 'standard', or 'minmax'")

        # fit on train, transform both
        X_train_np = X_train.numpy()
        X_valid_np = X_valid.numpy()

        scaler.fit(X_train_np)

        X_train = torch.tensor(scaler.transform(X_train_np), dtype=torch.float32)
        X_valid = torch.tensor(scaler.transform(X_valid_np), dtype=torch.float32)

    # ------------------------------------------------------
    # BUILD TORCH DATASETS + LOADERS
    # ------------------------------------------------------
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    if valid_batch_size is None:
        valid_batch_size = batch_size

    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)

    if return_scaler:
        return train_loader, valid_loader, scaler

    return train_loader, valid_loader

import math
from pathlib import Path
import sys

import numpy as np
import torch


def build_dataset(npy_dir):
    data = None
    for npy_file in Path(npy_dir).glob('*.npy'):
        if data is None:
            data = np.load(npy_file, allow_pickle=True)
        else:
            data = np.concatenate((data, np.load(npy_file, allow_pickle=True)))
    return data


def print_shapes(p_data, n_data):
    print(f"positive data shape: {p_data.shape}")
    print(f"negative data shape: {n_data.shape}")


def write_tensor(arr, filename):
    tensor = torch.from_numpy(arr).float()
    torch.save(tensor, filename)


def main():
    npy_dir = sys.argv[1]
    print(npy_dir)

    # Build the datasets

    p_data = build_dataset(f"{npy_dir}/positive")
    n_data = build_dataset(f"{npy_dir}/negative")
    print_shapes(p_data, n_data)

    # Balance the positive and negative sets

    p_len = len(p_data)
    n_len = len(n_data)
    if p_len > n_len:
        p_data = p_data[:n_len]
    elif n_len > p_len:
        n_data = n_data[:p_len]
    print_shapes(p_data, n_data)

    # If we are creating the training set, split into train and val

    name = npy_dir.split("/")[-1]
    if name == "train":
        np.random.shuffle(p_data)
        np.random.shuffle(n_data)

        train_size = math.ceil(0.8 * len(p_data))
        p_train = p_data[:train_size]
        p_val = p_data[train_size:]
        n_train = n_data[:train_size]
        n_val = n_data[train_size:]

        write_tensor(p_train, "train_positive.pt")
        write_tensor(p_val, "val_positive.pt")
        write_tensor(n_train, "train_negative.pt")
        write_tensor(n_val, "val_negative.pt")

    else:
        write_tensor(p_data, f"test_positive.pt")
        write_tensor(n_data, f"test_negative.pt")


if __name__ == "__main__":
    main()
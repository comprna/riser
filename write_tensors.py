import math
from pathlib import Path
import sys

import numpy as np
import torch


def build_dataset(npy_dir):
    data = None
    for npy_file in Path(npy_dir).glob('*.npy'):
        print(f"Loading {npy_file}...")
        if data is None:
            data = np.load(npy_file, allow_pickle=True)
        else:
            data = np.concatenate((data, np.load(npy_file, allow_pickle=True)))
    return data


def print_shapes(c_data, n_data):
    print(f"Coding data shape: {c_data.shape}")
    print(f"Noncoding data shape: {n_data.shape}")


def write_tensor(arr, filename):
    tensor = torch.from_numpy(arr).float()
    torch.save(tensor, filename)


def main():
    npy_dir = sys.argv[1]
    print(npy_dir)

    # Build the datasets

    c_data = build_dataset(f"{npy_dir}/coding")
    n_data = build_dataset(f"{npy_dir}/noncoding")
    print_shapes(c_data, n_data)

    # Balance the coding and noncoding sets

    c_len = len(c_data)
    n_len = len(n_data)
    if c_len > n_len:
        c_data = c_data[:n_len]
    elif n_len > c_len:
        n_data = n_data[:c_len]
    print_shapes(c_data, n_data)

    # If we are creating the training set, split into train and val

    name = npy_dir.split("/")[-1]
    if name == "train":
        np.random.shuffle(c_data)
        np.random.shuffle(n_data)

        train_size = math.ceil(0.8 * len(c_data))
        c_train = c_data[:train_size]
        c_val = c_data[train_size:]
        n_train = n_data[:train_size]
        n_val = n_data[train_size:]

        write_tensor(c_train, "train_coding.pt")
        write_tensor(c_val, "val_coding.pt")
        write_tensor(n_train, "train_noncoding.pt")
        write_tensor(n_val, "val_noncoding.pt")

    else:
        write_tensor(c_data, f"test_coding.pt")
        write_tensor(n_data, f"test_noncoding.pt")


if __name__ == "__main__":
    main()
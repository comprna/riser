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


def print_shapes(c_data, n_data):
    print(f"Coding data shape: {c_data.shape}")
    print(f"Noncoding data shape: {n_data.shape}")


def write_tensor(arr, filename):
    tensor = torch.from_numpy(arr)
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

    # Convert to tensors and write to file

    write_tensor(c_data, f"{sys.argv[2]}_coding.pt")
    write_tensor(n_data, f"{sys.argv[2]}_noncoding.pt")


if __name__ == "__main__":
    main()
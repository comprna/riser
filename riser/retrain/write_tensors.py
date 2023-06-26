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
    print("Dataset sizes before balancing:")
    print_shapes(p_data, n_data)

    # Balance the positive and negative sets

    p_len = len(p_data)
    n_len = len(n_data)
    if p_len > n_len:
        p_data = p_data[:n_len]
    elif n_len > p_len:
        n_data = n_data[:p_len]
    print("Dataset sizes after balancing positive and negative sets:")
    print_shapes(p_data, n_data)

    # Write tensors

    dataset = npy_dir.split("/")[-1]
    write_tensor(p_data, f"{dataset}_positive.pt")
    write_tensor(n_data, f"{dataset}_negative.pt")


if __name__ == "__main__":
    main()
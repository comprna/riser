from pathlib import Path
import sys

import numpy as np
import torch


def build_dataset(npy_dir):
    data = None
    for npy_file in Path(npy_dir).glob('*.npy'):
        print(npy_file)
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
    pos_npy_file = sys.argv[1]
    neg_npy_file = sys.argv[2]
    out_dir = sys.argv[3]

    # Build the datasets

    print(pos_npy_file)
    print(neg_npy_file)
    p_data = np.load(pos_npy_file, allow_pickle=True)
    n_data = np.load(neg_npy_file, allow_pickle=True)
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

    write_tensor(p_data, f"{out_dir}/positive.pt")
    write_tensor(n_data, f"{out_dir}/negative.pt")


if __name__ == "__main__":
    main()
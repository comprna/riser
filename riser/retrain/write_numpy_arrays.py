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

    # Create labels

    p_labels = torch.ones(p_data.shape[0], dtype=torch.long)
    n_labels = torch.zeros(n_data.shape[0], dtype=torch.long)

    # Merge positive and negative sets

    data = np.concatenate((p_data, n_data), axis=0)
    labels = np.concatenate((p_labels, n_labels), axis=0)

    # Write arrays to file

    np.save(f"{out_dir}/data.npy", data)
    np.save(f"{out_dir}/labels.npy", labels)


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    npy_dir = sys.argv[1]

    # Concatenate all numpy files comprising the dataset

    data = None
    for npy_file in Path(npy_dir).glob('*.npy'):
        if data is None:
            data = np.load(npy_file, allow_pickle=True)
        else:
            data = np.concatenate((data, np.load(npy_file, allow_pickle=True)))

    # Convert numpy to tensor and write to file

    data = torch.from_numpy(data)
    torch.save(data, f"{sys.argv[2]}.pt")


if __name__ == "__main__":
    main()
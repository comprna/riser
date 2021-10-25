from pathlib import Path
import sys

import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file


def mad_normalise(signal, outlier_lim):
    if signal.shape[0] == 0:
        raise ValueError("Signal must not be empty")
    median = np.median(signal)
    mad = calculate_mad(signal, median)
    vnormalise = np.vectorize(normalise)
    normalised = vnormalise(np.array(signal), median, mad)
    return smooth_outliers(normalised, outlier_lim)


def smooth_outliers(arr, outlier_lim):
    # Replace outliers with average of neighbours
    outlier_idx = np.asarray(np.abs(arr) > outlier_lim).nonzero()[0]
    for i in outlier_idx:
        if i == 0:
            arr[i] = arr[i+1]
        elif i == len(arr)-1:
            arr[i] = arr[i-1]
        else:
            arr[i] = (arr[i-1] + arr[i+1])/2
            # Clip any outliers that still remain after smoothing
            if arr[i] > outlier_lim:
                arr[i] = outlier_lim
            elif arr[i] < -1 * outlier_lim:
                arr[i] = -1 * outlier_lim
    return arr


def calculate_mad(signal, median):
    f = lambda x, median: np.abs(x - median)
    distances_from_median = f(signal, median)
    return np.median(distances_from_median)


def normalise(x, median, mad):
    return (x - median) / (1.4826 * mad)


def main():

    # Number of samples to use per signal

    n_secs = int(sys.argv[1])
    freq   = 3012
    cutoff = freq * n_secs

    # Location of raw signals

    f5_dir = sys.argv[2]
    name = f5_dir.split("/")[-1]

    # Store processed data

    data = []

    # Iterate through files

    for f5_file in Path(f5_dir).glob('*.fast5'):

        # Iterate through signals in file

        n_discarded = 0
        print(f"Processing {f5_file}...")
        with get_fast5_file(f5_file, mode="r") as f5:
            for i, read in enumerate(f5.get_reads()):

                # Retrieve raw current measurements

                signal_pA = read.get_raw_data(scale=True)

                # Keep first N signal values

                if len(signal_pA) < cutoff:
                    n_discarded += 1
                    continue
                signal_pA = signal_pA[:cutoff]

                # Normalise signal

                outlier_lim = 3.5
                normalised = mad_normalise(signal_pA, outlier_lim)
                data.append(normalised)

        print(f"# of discarded reads (< {cutoff} samples) in {f5_file}: {n_discarded}")


    # Write data to file

    np.save(f"{name}_{cutoff}.npy", np.array(data))


if __name__=="__main__":
    main()

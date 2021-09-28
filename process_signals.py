import h5py

from matplotlib import pyplot as plt
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file


def mad_normalise(signal, outlier_z_score):
    if signal.shape[0] == 0:
        raise ValueError("Signal must not be empty to normalise")
    median = np.median(signal)
    mad = calculate_mad(signal, median)
    vnormalise_value = np.vectorize(normalise_value)
    return vnormalise_value(np.array(signal), median, mad, outlier_z_score)

def calculate_mad(signal, median):
    f = lambda x, median: np.abs(x - median)
    distances_from_median = f(signal, median)
    return np.median(distances_from_median)

def normalise_value(x, median, mad, outlier_z_score):
    modified_z_score = calculate_modified_z_score(x, median, mad)
    if modified_z_score > outlier_z_score:
        return outlier_z_score
    elif modified_z_score < -1 * outlier_z_score:
        return -1 * outlier_z_score
    else:
        return modified_z_score 

def calculate_modified_z_score(x, median, mad):
    return (x - median) / (1.4826 * mad)


def main():

    # Number of samples to use per signal

    n_secs = 3
    freq   = 3012
    cutoff = freq * n_secs

    # Iterate through fast5 files for training set

    fast5_file = "/home/alex/OneDrive/phd-project/rna-classifier/4_BoostNanoSegmentSignals/hek293_test_coding/hek293_test_coding1.fast5"

    # Iterate through signals in fast5 file

    n_discarded = 0

    with get_fast5_file(fast5_file, mode="r") as f5:
        for read in f5.get_reads():
            signal_pA = read.get_raw_data(scale=True)
            # signal_pA = signal_pA[:cutoff]
            outlier_z_score = 4 # TODO: Determine how to deal with outliers
            normalised_pA = mad_normalise(signal_pA, outlier_z_score)
            break


    with h5py.File(fast5_file, 'r') as h5:
        for read in h5.keys():

            # Retrieve signal using ONT fast5 API to get raw current
            # measurements, not ADC values from fast5 file

            signal = h5[f"{read}/Raw/Signal"][()] # TODO: Update

            # Use first N signal values

            # if len(signal) < cutoff:
            #     n_discarded += 1
            #     continue
            
            # signal = signal[:cutoff]

            # Normalise signal

            outlier_z_score = 4 # TODO: Determine how to deal with outliers
            normalised = mad_normalise(signal, outlier_z_score)

            # Plot signal to check normalisation worked

            # _, axs = plt.subplots(2, 1, sharex="all")
            # axs[0].plot(signal)
            # axs[1].plot(normalised)
            # plt.show()
        
            break

        _, axs = plt.subplots(2, 2, sharex="all")
        axs[0][0].plot(signal)
        axs[0][0].set_title("ADC signal")
        axs[1][0].plot(signal_pA)
        axs[1][0].set_title("Raw current")
        axs[0][1].plot(normalised)
        axs[0][1].set_title("Normalised ADC signal")
        axs[1][1].plot(normalised_pA)
        axs[1][1].set_title("Normalised raw current")
        plt.show()
            


if __name__=="__main__":
    main()

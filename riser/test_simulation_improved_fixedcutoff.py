from pathlib import Path
import sys

from attrdict import AttrDict
from matplotlib import pyplot as plt
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import torch
from torchinfo import summary
import yaml

from nets.cnn import ConvNet

OUTLIER_LIMIT = 3.5
SCALING_FACTOR = 1.4826
SAMPLING_HZ = 3012
FIXED_CUTOFF = 6481
MAX_INPUT_LENGTH = 4

def get_config(filepath):
    with open(filepath) as config_file:
        return AttrDict(yaml.load(config_file, Loader=yaml.Loader))


def classify(signal, device, model):
    with torch.no_grad():
        X = torch.from_numpy(signal).unsqueeze(0)
        X = X.to(device, dtype=torch.float)
        logits = model(X)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs

def mad_normalise(signal):
    if signal.shape[0] == 0:
        raise ValueError("Signal must not be empty")
    median = np.median(signal)
    mad = calculate_mad(signal, median)
    vnormalise = np.vectorize(normalise)
    normalised = vnormalise(np.array(signal), median, mad)
    return smooth_outliers(normalised)

def calculate_mad(signal, median):
    f = lambda x, median: np.abs(x - median)
    distances_from_median = f(signal, median)
    return np.median(distances_from_median)

def normalise(x, median, mad):
    # TODO: Handle divide by zero
    return (x - median) / (SCALING_FACTOR * mad)

def smooth_outliers(arr):
    # Replace outliers with average of neighbours
    outlier_idx = np.asarray(np.abs(arr) > OUTLIER_LIMIT).nonzero()[0]
    for i in outlier_idx:
        if i == 0:
            arr[i] = arr[i+1]
        elif i == len(arr)-1:
            arr[i] = arr[i-1]
        else:
            arr[i] = (arr[i-1] + arr[i+1])/2
            # Clip any outliers that still remain after smoothing
            arr[i] = clip_if_outlier(arr[i])
    return arr

def clip_if_outlier(x):
    if x > OUTLIER_LIMIT:
        return OUTLIER_LIMIT
    elif x < -1 * OUTLIER_LIMIT:
        return -1 * OUTLIER_LIMIT
    else:
        return x

def get_polyA_coords(signal, resolution, mad_threshold):
    # plt.figure(figsize=(12,6))
    # plt.plot(signal)
    i = 0
    polyA_start = None
    polyA_end = None
    history = 2 * resolution
    while i + resolution <= len(signal):
        # Calculate median absolute deviation of this window
        median = np.median(signal[i:i+resolution])
        mad = calculate_mad(signal[i:i+resolution], median)

        # Calculate percentage change of mean for this window
        mean = np.mean(signal[i:i+resolution])
        rolling_mean = mean
        if i > history:
            rolling_mean = np.mean(signal[i-history:i])
        mean_change = (mean - rolling_mean) / rolling_mean * 100

        # Start condition
        if not polyA_start and mean_change > 20 and mad <= mad_threshold:
            polyA_start = i

        # End condition
        if polyA_start and not polyA_end and mad > 20:
            polyA_end = i
        
        # plt.axvline(i+resolution, color='red')
        # plt.text(i+resolution, 500, int(mad))
        # plt.text(i+resolution, 900, int(mean_change))
        i += resolution

    # if polyA_start: plt.axvline(polyA_start, color='green')
    # if polyA_end: plt.axvline(polyA_end, color='green')
    # plt.savefig(f"{read_id}_{polyA_start}_{polyA_end}.png")
    # plt.clf()

    return polyA_start, polyA_end

def main():
    # Location of raw signals
    f5_dir = sys.argv[1]
    dataset = f5_dir.split("/")[-1]

    # Setup
    model_file = sys.argv[2]
    config_file = sys.argv[3]
    resolution = int(sys.argv[4])
    mad_threshold = int(sys.argv[5])

    # Load config
    config = get_config(config_file)

    # Test info
    model_id = model_file.split('.pth')[0].split('/')[-1]

    # Get device for model evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Define model
    model = ConvNet(config.cnn).to(device)
    model.load_state_dict(torch.load(model_file))
    summary(model)
    model.eval()

    # Iterate through files
    for f5_file in Path(f5_dir).glob('*.fast5'):
        filename = f5_file.name.split("/")[-1]

        # Iterate through signals in file
        with get_fast5_file(f5_file, mode="r") as f5:
            for i, read in enumerate(f5.get_reads()):
                preds = {}

                # Retrieve raw current measurements
                orig_signal_pA = read.get_raw_data(scale=False)

                # Simulate retrieving data in 1s chunks
                j = 1
                polyA_start = None
                polyA_end = None
                keep_going = True
                while len(orig_signal_pA) >= j * SAMPLING_HZ and keep_going:
                    signal_pA = orig_signal_pA[:SAMPLING_HZ * j]

                    # Find the polyA if we haven't already for this read
                    if polyA_start is None or polyA_end is None:
                        polyA_start, polyA_end = get_polyA_coords(signal_pA, resolution, mad_threshold)
                    
                    # If we haven't found the polyA yet
                    if polyA_end is None:
                        
                        # And a certain time has passed, then just revert
                        # to a fixed cutoff approach
                        if len(signal_pA) > FIXED_CUTOFF + 4 * SAMPLING_HZ:
                            # Cutoff polyA using fixed approach
                            signal_pA = signal_pA[FIXED_CUTOFF:]

                            # Input max length 4s to network
                            signal_pA = signal_pA[:4 * SAMPLING_HZ]

                        # Otherwise, try again with the next chunk
                        else:
                            preds[j] = "no_polyA\tno_polyA"
                            j += 1
                            continue

                    # If we have found the polyA then cutoff
                    if polyA_end is not None:
                        signal_pA = signal_pA[polyA_end+1:]

                        # Can only consider transcript signals at least 2s long
                        min_length = 2 * SAMPLING_HZ
                        if len(signal_pA) < min_length:
                            preds[j] = "too_short\ttoo_short"
                            j += 1
                            continue

                        # Signal input to network can be max 4s long
                        max_length = 4 * SAMPLING_HZ
                        if len(signal_pA) > max_length:
                            signal_pA = signal_pA[:max_length]
                            keep_going = False

                    # Normalise
                    normalised = mad_normalise(signal_pA)
                    
                    # Classify
                    probs = classify(normalised, device, model)
                    prob_n = probs[0][0].item()
                    prob_p = probs[0][1].item()

                    preds[j] = f"{prob_n}\t{prob_p}"

                    j += 1
                
                print(f"PRED\t{model_id}\t{dataset}\t{filename}\t{read.read_id}\t{polyA_start}\t{polyA_end}\t{preds}\n")


if __name__ == "__main__":
    main()
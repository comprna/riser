from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import torch
from torchinfo import summary

from nets.cnn import ConvNet
from nets.resnet import ResNet
from nets.tcn import TCN
from utilities import get_config

OUTLIER_LIMIT = 3.5
SCALING_FACTOR = 1.4826
SAMPLING_HZ = 3012

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

def get_polyA_coords(signal):
    i = 0
    polyA_start = None
    polyA_end = None
    resolution = 500
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
        if not polyA_start and mean_change > 20 and mad <= 15:
            polyA_start = i

        # End condition
        if polyA_start and not polyA_end and mad > 20:
            polyA_end = i + resolution
        i += resolution

    plt.axvline(polyA_start, color='green')
    plt.axvline(polyA_end, color='green')
    plt.show()
    plt.save(f"{polyA_start}_{polyA_end}.png")

    return polyA_start, polyA_end

def main():
    # Location of raw signals
    f5_dir = sys.argv[1]
    dataset = f5_dir.split("/")[-1]

    # Setup
    model_file = sys.argv[2]
    config_file = sys.argv[3]

    # Have the signals already been trimmed by BoostNano?
    already_trimmed = sys.argv[4]
    if already_trimmed == "Y":
        already_trimmed = True
    elif already_trimmed == "N":
        already_trimmed = False
    else:
        print(f"already_trimmed value {already_trimmed} invalid!")
        exit()
    
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

                # Retrieve raw current measurements
                signal_pA = read.get_raw_data(scale=True)

                # If needed, trim sequencing adapter & polyA with dynamic cutoff
                polyA_start = "boostnano"
                polyA_end = "boostnano"
                if not already_trimmed:
                    polyA_start, polyA_end = get_polyA_coords(signal_pA)

                    # If polyA start or end is none
                    if polyA_start is None or polyA_end is None:
                        print(f"PRED\t{model_id}\t{dataset}\t{filename}\t{read.read_id}\t{polyA_start}\t{polyA_end}\tNA\tNA\tNA\tNA\tNA\tNA\n")
                        continue

                    # Otherwise, trim
                    signal_pA = signal_pA[polyA_end+1:]

                # Predict for each incremental input signal length
                preds = {}
                for j in range(2,5): # 2,3,4
                    # If the signal isn't long enough
                    cutoff = SAMPLING_HZ * j
                    if len(signal_pA) < cutoff:
                        preds[j] = f"NA\tNA"
                        continue

                    # Trim to input length
                    trimmed = signal_pA[:cutoff]

                    # Normalise signal
                    normalised = mad_normalise(trimmed)

                    # Predict
                    probs = classify(normalised, device, model)
                    prob_n = probs[0][0].item()
                    prob_p = probs[0][1].item()

                    preds[j] = f"{prob_n}\t{prob_p}"
                
                print(f"PRED\t{model_id}\t{dataset}\t{filename}\t{read.read_id}\t{polyA_start}\t{polyA_end}\t{preds[2]}\t{preds[3]}\t{preds[4]}\n")


if __name__ == "__main__":
    main()
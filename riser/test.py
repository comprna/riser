import math
from pathlib import Path
import sys

import attridict
from matplotlib import pyplot as plt
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import torch
from torchinfo import summary
import yaml

from nets.cnn import ConvNet


OUTLIER_LIMIT = 3.5
SCALING_FACTOR = 1.4826
SAMPLING_HZ_RNA002 = 3012
SAMPLING_HZ_RNA004 = 4000
MIN_SIGNAL_LENGTH = 4096 # Min input length that CNN can handle
MIN_SIGNAL_SEC_RNA002 = MIN_SIGNAL_LENGTH / SAMPLING_HZ_RNA002
MIN_SIGNAL_SEC_RNA004 = MIN_SIGNAL_LENGTH / SAMPLING_HZ_RNA004
MAX_SIGNAL_SEC_RNA002 = 4
MAX_SIGNAL_SEC_RNA004 = 2.15 # 280nt @ 130bps (make decision after same # nt passed as 002)
FIXED_TRIM_RNA002 = 6481
FIXED_TRIM_RNA004 = 4634

def get_config(filepath):
    with open(filepath) as config_file:
        return attridict(yaml.load(config_file, Loader=yaml.Loader))

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
    sqk_kit = sys.argv[4]

    # Where to write test output
    out_dir = sys.argv[5]

    # Set up sequencing kit parameters
    if sqk_kit == "RNA002":
        sampling_hz = SAMPLING_HZ_RNA002
        min_sec = MIN_SIGNAL_SEC_RNA002
        max_sec = MAX_SIGNAL_SEC_RNA002
        fixed_trim = FIXED_TRIM_RNA002
    elif sqk_kit == "RNA004":
        sampling_hz = SAMPLING_HZ_RNA004
        min_sec = MIN_SIGNAL_SEC_RNA004
        max_sec = MAX_SIGNAL_SEC_RNA004
        fixed_trim = FIXED_TRIM_RNA004
    else:
        print(f"Sequencing kit invalid")
        exit()

    # Have the signals already been trimmed by BoostNano?
    already_trimmed = sys.argv[6]
    if already_trimmed == "Y":
        already_trimmed = True
    elif already_trimmed == "N":
        already_trimmed = False
        resolution = int(sys.argv[7])
        mad_threshold = int(sys.argv[8])
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
    for f5_file in Path(f5_dir).glob('**/*.fast5'):
        filename = f5_file.name.split("/")[-1]
        out = []

        # Iterate through signals in file
        with get_fast5_file(f5_file, mode="r") as f5:
            for i, read in enumerate(f5.get_reads()):

                # Retrieve raw current measurements
                signal_pA = read.get_raw_data(scale=False)

                # If needed, trim sequencing adapter & polyA with dynamic cutoff
                polyA_start = "boostnano"
                polyA_end = "boostnano"
                if not already_trimmed:
                    polyA_start, polyA_end = get_polyA_coords(signal_pA, resolution, mad_threshold)

                    # If polyA start or end is none, couldn't find polyA so
                    # use fixed trim length. Otherwise, trim with computed length.
                    if polyA_end:
                        signal_pA = signal_pA[polyA_end+1:]
                    else:
                        signal_pA = signal_pA[fixed_trim:]

                # Predict for each incremental input signal length
                preds = {}
                input_length = math.ceil(min_sec * sampling_hz)
                max_input_length = math.floor(max_sec * sampling_hz)
                while input_length <= max_input_length:
                    # If the signal isn't long enough
                    if len(signal_pA) < input_length:
                        input_length += sampling_hz
                        continue

                    # Trim to input length
                    trimmed = signal_pA[:input_length]

                    # Normalise signal
                    normalised = mad_normalise(trimmed)

                    # Predict
                    probs = classify(normalised, device, model)
                    prob_n = probs[0][0].item()
                    prob_p = probs[0][1].item()

                    preds[input_length] = f"{input_length}:{prob_n},{prob_p}"

                    # Increment by 1s
                    input_length += sampling_hz
                
                out.append(f"{model_id}\t{dataset}\t{filename}\t{read.read_id}\t{polyA_start}\t{polyA_end}\t{';'.join([preds[x] for x in preds.keys()])}\n")

        # Write output for this fast5 file
        with open(f"{out_dir}/{filename}_test_output.tsv", "w") as out_f:
            for line in out:
                out_f.write(line)

if __name__ == "__main__":
    main()
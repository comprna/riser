import numpy as np

from matplotlib import pyplot as plt


_OUTLIER_LIMIT = 3.5
_SCALING_FACTOR = 1.4826
_SAMPLING_HZ = 3012
_TRIM_RESOLUTION = 500
_TRIM_MAD_THRESHOLD = 20

class SignalProcessor():
    def __init__(self, min_input_s, max_input_s):
        self.min_txt_length = min_input_s * _SAMPLING_HZ
        self.max_txt_length = max_input_s * _SAMPLING_HZ

    def is_max_length(self, signal):
        return len(signal) >= self.max_txt_length

    def preprocess(self, signal):
        # If signal is shorter than min length then pad after normalising
        if len(signal) < self.min_txt_length:
            signal = self.mad_normalise(signal)
            pad_len = self.min_txt_length - len(signal)
            signal = np.pad(signal, ((pad_len, 0)), constant_values=(0,))
        # If signal is longer than max length then trim before normalising
        elif self.is_max_length(signal):
            signal = signal[:self.max_txt_length]
            signal = self.mad_normalise(signal)
        return signal

    def get_polyA_end(self, signal):
        # plt.figure(figsize=(12,6))
        # plt.plot(signal)
        i = 0
        polyA_start = None
        polyA_end = None
        history = 2 * _TRIM_RESOLUTION
        while i + _TRIM_RESOLUTION <= len(signal):
            # Calculate median absolute deviation of this window
            median = np.median(signal[i:i+_TRIM_RESOLUTION])
            mad = self._calculate_mad(signal[i:i+_TRIM_RESOLUTION], median)

            # Calculate percentage change of mean for this window
            mean = np.mean(signal[i:i+_TRIM_RESOLUTION])
            rolling_mean = mean
            if i > history:
                rolling_mean = np.mean(signal[i-history:i])
            mean_change = (mean - rolling_mean) / rolling_mean * 100

            # Start condition
            if not polyA_start and mean_change > 20 and mad <= _TRIM_MAD_THRESHOLD:
                polyA_start = i

            # End condition
            if polyA_start and not polyA_end and mad > 20:
                polyA_end = i
            
            # plt.axvline(i+_TRIM_RESOLUTION, color='red')
            # plt.text(i+_TRIM_RESOLUTION, 500, int(mad))
            # plt.text(i+_TRIM_RESOLUTION, 900, int(mean_change))
            i += _TRIM_RESOLUTION

        # if polyA_start: plt.axvline(polyA_start, color='green')
        # if polyA_end: plt.axvline(polyA_end, color='orange')
        # plt.savefig(f"{read_id}_{polyA_start}_{polyA_end}.png")
        # plt.clf()

        return polyA_end

    def trim_polyA(self, signal, read_id, cache):
        """
        If the polyA end can be found, then trim polyA + sequencing adapter 
        from start of signal.
        """
        trimmed = False
        if read_id in cache:
            polyA_end = cache[read_id]
        else:
            polyA_end = self.get_polyA_end(signal)
            if polyA_end:
                cache[read_id] = polyA_end
        if polyA_end:
            signal = signal[polyA_end+1:]
            trimmed = True
        return signal, trimmed

    def mad_normalise(self, signal):
        if signal.shape[0] == 0:
            raise ValueError("Signal must not be empty")
        median = np.median(signal)
        mad = self._calculate_mad(signal, median)
        vnormalise = np.vectorize(self._normalise)
        normalised = vnormalise(np.array(signal), median, mad)
        return self._smooth_outliers(normalised)

    def _calculate_mad(self, signal, median):
        f = lambda x, median: np.abs(x - median)
        distances_from_median = f(signal, median)
        return np.median(distances_from_median)

    def _normalise(self, x, median, mad):
        # TODO: Handle divide by zero
        return (x - median) / (_SCALING_FACTOR * mad)

    def _smooth_outliers(self, arr):
        # Replace outliers with average of neighbours
        outlier_idx = np.asarray(np.abs(arr) > _OUTLIER_LIMIT).nonzero()[0]
        for i in outlier_idx:
            if i == 0:
                arr[i] = arr[i+1]
            elif i == len(arr)-1:
                arr[i] = arr[i-1]
            else:
                arr[i] = (arr[i-1] + arr[i+1])/2
                # Clip any outliers that still remain after smoothing
                arr[i] = self._clip_if_outlier(arr[i])
        return arr

    def _clip_if_outlier(self, x):
        if x > _OUTLIER_LIMIT:
            return _OUTLIER_LIMIT
        elif x < -1 * _OUTLIER_LIMIT:
            return -1 * _OUTLIER_LIMIT
        else:
            return x


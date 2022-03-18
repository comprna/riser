import numpy as np
from matplotlib import pyplot as plt

_OUTLIER_LIMIT = 3.5
_SCALING_FACTOR = 1.4826

class SignalProcessor():
    def __init__(self, polya_length, input_length):
        self.polya_length = polya_length
        self.input_length = input_length

    def process(self, signal):
        """
        Trim polyA + sequencing adapter from start of signal
        Retain the first 4 seconds of transcript signal
        Normalise
        """
        signal = signal[self.polya_length:]
        signal = signal[:self.input_length]
        return self._mad_normalise(signal)

    def get_min_length(self):
        return self.polya_length + self.input_length

    def _mad_normalise(self, signal):
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


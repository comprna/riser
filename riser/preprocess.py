import numpy as np

_OUTLIER_LIMIT = 3.5
_SCALING_FACTOR = 1.4826
_SAMPLING_HZ = 3012

class SignalProcessor():
    def __init__(self, trim_length, min_input_s, max_input_s):
        self.trim_length = trim_length # TODO: rename trim_length
        self.min_txt_length = min_input_s * _SAMPLING_HZ
        self.max_txt_length = max_input_s * _SAMPLING_HZ

    def trim_polyA(self, signal):
        """
        Trim polyA + sequencing adapter from start of signal
        using fixed cutoff amount.
        """
        return signal[self.trim_length:]

    def mad_normalise(self, signal):
        if signal.shape[0] == 0:
            raise ValueError("Signal must not be empty")
        median = np.median(signal)
        mad = self._calculate_mad(signal, median)
        vnormalise = np.vectorize(self._normalise)
        normalised = vnormalise(np.array(signal), median, mad)
        return self._smooth_outliers(normalised)

    def get_min_assessable_length(self):
        return self.trim_length + self.min_txt_length

    def get_max_assessable_length(self):
        return self.trim_length + self.max_txt_length

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


import numpy as np


class MinMaxScaler:
    """Min Max normalizer.
    Args:
    - data: original data
    Returns:
    - norm_data: normalized data
    """

    def __init__(self, by_axis, eps=1e-16):
        self.by_axis = by_axis
        self.eps = eps

    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        # Axis where shape is being subtracted
        axis = int((1-self.by_axis))+1
        # Get transformation params and reshape accordingly
        shape = np.ones(len(np.shape(data)), dtype=int)
        shape[axis] = int(data.shape[axis])
        shape = tuple(shape)

        self.mini = np.reshape(np.min(np.min(data, 0), axis=self.by_axis), shape)
        self.range = np.reshape(np.max(np.max(data, 0), axis=self.by_axis), shape) - self.mini

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + self.eps)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data
    
      



import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=10):
    num_hours = 30 * 24 * 12
    X_satellite = np.random.rand(num_samples, 128, 128, 3)
    X_temporal = np.random.rand(num_samples, num_hours, 1)
    X_weather = np.random.rand(num_samples, 10)
    X_soil = np.random.rand(num_samples, 5)
    X_crop_info = np.random.rand(num_samples, 7)
    Y = np.random.randint(2, size=(num_samples, 1))
    return X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y

def dataframe_from_array(data, prefix='Feature'):
    return pd.DataFrame(data, columns=[f'{prefix}_{i + 1}' for i in range(data.shape[1])])

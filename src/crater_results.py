import pandas as pd
import numpy as np

craters = pd.read_csv('crater_logs_noisy.csv', header=0)

# print(craters[np.isnan(craters.iloc[600:,4].values)])
print(craters.iloc[600:,:][False == np.isnan(craters.iloc[600:,4].values)].index)
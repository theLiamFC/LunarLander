import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_crater_count(crater_log_fname):
    craters = pd.read_csv(crater_log_fname, header=0)

    crater_count = []

    for row in range(craters.shape[0]):
        count = 0
        for i in range(4,len(craters.columns),3):
            if not np.isnan(craters.iloc[row,i]):
                count+=1
            else:
                break
        crater_count.append(count)

    return crater_count
import pandas as pd

file = pd.read_csv("crater_logs.csv", header=0).iloc[::-1,:]
print(file.values[0,:].flatten())
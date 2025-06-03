import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

craters = pd.read_csv('crater_logs.csv', header=0)

crater_count = []

for row in range(craters.shape[0]):
    count = 0
    for i in range(4,len(craters.columns),3):
        if not np.isnan(craters.iloc[row,i]):
            count+=1
        else:
            break
    crater_count.append(count)
# print(crater_count)

t = np.arange(0,craters.shape[0], 1)

plt.scatter(t, crater_count)
plt.title('Crater Count vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Crater Count')
plt.show()
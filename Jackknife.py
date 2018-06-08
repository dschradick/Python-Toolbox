########## JACKKNIFE
import numpy as np

heights = np.random.normal(size=1000,loc=179,scale=20)
mean_lengths, n = [], len(heights)
index = np.arange(n)

for i in range(n):
    jk_sample = heights[index != i]
    mean_lengths.append(np.mean(jk_sample))

mean_lengths = np.array(mean_lengths)
print("Jackknife estimate of the mean = {}".format(np.mean(mean_lengths)))

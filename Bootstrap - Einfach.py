########## BOOTRAPPING - EINFACH
import numpy as np
import pandas as pd

heights = sorted(np.random.normal(size=1000,loc=179,scale=20))
weights = sorted(np.random.normal(size=1000,loc=60,scale=10))
df = pd.DataFrame({'heights':heights,'weights':weights})

sims, data_size, height_medians, hw_corr = 100, df.shape[0], [], []

for i in range(sims):
    tmp_df = df.sample(n=data_size, replace=True)
    height_medians.append(tmp_df['heights'].median())
    hw_corr.append(tmp_df.weights.corr(tmp_df.heights))

median_ci = np.percentile(height_medians, [2.5, 97.5])
correlation_ci = np.percentile(hw_corr, [2.5, 97.5])
print("Height Median CI = {} \nHeight Weight Correlation CI = {}".format(median_ci,correlation_ci))

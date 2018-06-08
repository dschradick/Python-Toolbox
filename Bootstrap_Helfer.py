import numpy as np

## Empirische Verteilungsfunktion
# Häufig besser zum Verteilungen zu vergleichen als Histogramm (wg. binning-bias)
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y


# Erzeugt bootstrap sample
# => von draw_bs_reps benutzt um bootstrap sample zu erzeugen
def bootstrap_replicate_1d(data,func):
    bs_sample = np.random.choice(data,len(data),replace=True)
    return func(bs_sample)

# Erzeugt mehrere boostrap-replicates
def draw_bs_reps(data, func, size=1):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)
    return bs_replicates


# Erzeugt bootstrap pairs für Lineare Regression
def draw_bs_pairs_linreg(x, y, size=1):
    inds = np.arange(len(x))
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x,bs_y,1)
    return bs_slope_reps, bs_intercept_reps


# Erzeugt Permutations-Sample von zwei Datensätzen
# (von draw_perm_reps benutzt)
def permutation_sample(data1, data2):
    data = np.concatenate((data1,data2))
    permuted_data = np.random.permutation(data)
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]
    return perm_sample_1, perm_sample_2


# Erzeugt mehrere Permutations-Replicates
def draw_perm_reps(data_1, data_2, func, size=1):
    perm_replicates = np.empty(size)
    for i in range(size):
        # immer ein neues Permutations-Sample nehmen
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates

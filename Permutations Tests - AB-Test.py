########## PERMUTATIONS TESTS - A/B-Test
# A/B-Test mittle eines Permutationstest
#
# H0: Die Clickthrough-Rate wird vom Redesign nicht beeinflusst
# HA: Clickthrough-Rate für Variante B ist höher
# Signifikanz-level = 0.05
import numpy as np
import Bootstrap_Helfer as bh
import matplotlib.pyplot as plt

#### Daten erzeugen
site_visitors = 1000
users_A = 500
users_B = 500
clicks_A = 45
clicks_B  = 65


#### Tatsächliche Verteilungen
clickthrough_A = [True] * clicks_A + [False] * (users_A - clicks_A)
clickthrough_B = [True] * clicks_B + [False] * (users_B - clicks_B)


#### Teststatistik
# Differnz der Verhältnisse = diff_frac
def diff_frac(data_A,data_B):
    frac_A = np.sum(data_A) / len(data_A)
    frac_B = np.sum(data_B) / len(data_B)
    return frac_B - frac_A


#### Beobachteter Wert der Teststatistik
observed_diff_in_frac = diff_frac(clickthrough_A,clickthrough_B)   # beobachteter Wert der Teststatistik
observed_diff_in_frac


#### Daten unter der Null-Hypothese simulieren
perm_replicates = bh.draw_perm_reps(clickthrough_A,clickthrough_B,diff_frac,10000)
np.sum(perm_replicates)


#### Visualisieren der Sampling Distribution unter H0
critical = np.percentile(perm_replicates,95)
_ = plt.hist(perm_replicates, bins=20, normed=True);
_ = plt.axvline(critical, color='r', linestyle='dashed') ;
_ = plt.axvline(observed_diff_in_frac, color='k', linestyle='solid')


#### Test auswerten
p_value = np.sum(perm_replicates > observed_diff_in_frac) / 10000
print(p_value)  # 0.0168 < 0.05
# => Null-Hypothese ablehnen
# ===> Redesign beeinflusst CTR, wobei B eine höhere Clickthrough-Rate aufweist

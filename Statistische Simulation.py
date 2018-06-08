########## STASTISTISCHE SIMULATION
import pandas as pd
import numpy as np

#### Vorbereitungen
## Zufallsvariablen
np.random.seed(0)
lam, size = 5, 1000
samples = np.random.poisson(lam, size)
samples

#### Shuffling
x = [1,2,3,4,5,6,7,8,9]
np.random.shuffle(x)      # inplace


#### Simulation
# A. Statistische Modellierung
#  1. Möglichen Outcomes definieren
#  2. Wahrscheinlichkeit zuweisen
#  3. Verhältnis zwischen den Variablen definieren
# B. Random sampling & Analyse
#  1. Generieren von mehreren Outcomes durch wiederholtes sampling
#  2. Outcomes analysieren

## Beispiel: Zwei mal Münze werfen und vergleichen
# Zwei Zufallsvariablen A und B sind gleichverteilt
# A. Statistische Modellierung
possible_outcomes, probabilities = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
A = np.random.choice(possible_outcomes, size=1, p=probabilities)
B = np.random.choice(possible_outcomes, size=1, p=probabilities)
 # B. Random sampling & Analyse
sims, wins = 100, 0
for i in range(sims):
    A = np.random.choice(die, size=1, p=probabilities)
    B = np.random.choice(die, size=1, p=probabilities)
    if A == B:
        wins = wins + 1

print("In {} games, you win {} times".format(sims, wins))




#### Simulation für Business-Entscheidungen
# Anstatt Analytischer Lösung
#
# Idee:
# Experiminte mit verschiedenen Inputs(variieren) durchführen und Outputs analysieren
#
# Beispiel:
# Wie darf ein Lotto-Ticket kosten, damit es Sinn macht eines zu kaufen
# => Ticketkosten variieren und betrachten, wann erwarteter Payoff negativ

## Modell für eine Ziehung
lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000
chance_of_winning = 1/num_tickets
size = 2000
# Entweder man verliehrt =>  Resultat = bezahlt, das Ticket
# oder man gewinnt       =>  Resualt  = gewinnt Preis - Ticketpreis
payoffs = [-lottery_ticket_cost, grand_prize-lottery_ticket_cost]
probs = [1-chance_of_winning, chance_of_winning]

outcomes = np.random.choice(a=payoffs, size=size, p=probs, replace=True)
answer = outcomes.mean()
print("Average payoff from {} simulations = {}".format(size, answer))

## Simulation zur Berechung des Break-Even Punktes
sims, lottery_ticket_cost = 3000, 0

# inkrementieren von lottery_ticket_cost` bis mittlerer Wert des Outcomes unter 0 fällt
while 1:
    outcomes = np.random.choice([-lottery_ticket_cost, grand_prize-lottery_ticket_cost],
                 size=sims, p=[1-chance_of_winning, chance_of_winning], replace=True)
    if outcomes.mean() < 0:
        break
    else:
        lottery_ticket_cost += 1
answer = lottery_ticket_cost - 1

print("The highest price at which it makes sense to buy the ticket is {}".format(answer))





#### Daten generieren
# Daten generierenden Prozess bestimmen durch folgende drei Schritte:
# 1. Welche Faktoren beeinflussen die Daten
# 2. Quellen der Unsicherheit (hier: z.B. Wetter)
# 3. Beziehungen zwischen den Faktoren
# ==> als BAUM darstellen mit OUTCOME ALS WURZEL

## Beispiel: Fahrprüfung
# Wahrscheinlichkeit zu bestehen:
# - bei sonne: 90%
# - bei regen: 30%
# Wahrscheinlichkeit für Regen: 40%
sims, outcomes = 1000, []
p_rain, p_pass = 0.40, {'sun':0.9, 'rain':0.3}

# Explizite Wahrscheinlichenkeiten angeben
# np.random.choice(['rain', 'sun'], p=[p_rain, 1-p_rain])

def test_outcome(p_rain):
    # Simulieren ob es regnet
    weather = np.random.choice(['rain', 'sun'], p=[p_rain, 1-p_rain])
    # Simulieren ob bestanden
    return np.random.choice(['pass', 'fail'], p=[p_pass[weather], 1-p_pass[weather]])

for _ in range(sims):
    outcomes.append(test_outcome(p_rain))

# Simulieren, wie wahrscheinlich es ist, dass man besteht
prob_of_passing = sum([x == 'pass' for x in outcomes])/len(outcomes)*100
print("Probability of Passing the driving test = {prob:.2f}%".format(prob=prob_of_passing))


## Bei bekannten Wahrscheinlichkeiten für binomial verteilte Daten
# Beispiel US-Wahlen
# Jeder Staat hat eine gewisse Wahrscheinlichkeit, dass rot gewinnt
# => was ist wahrscheinlichkeit, dass rot weniger als 45% der Staaten gewinnt
p = np.random.normal(loc=0.4,scale=0.1)
outcomes, sims, probs = [], 1000, p

for _ in range(sims):
    # Simulate elections in the 50 states
    election = np.mean(np.random.binomial(p=probs, n=1))
    # Get average of Red wins and add to `outcomes`
    outcomes.append(election.mean())

print("Probability of Red winning in less than 45% of the states = {}".format(sum([(x < 0.45) for x in outcomes])/len(outcomes)))



## Komplexerer GDP: Fitness Goal
# Schritte: 15k vs 5k abhängig ob man zum Fitnessstudio geht
# Wahrscheinlichkeit für Fitness-Studio: 40
# Wenn > 10k Schritte: 80% => 1kg abnehmen; 20% 1kg zunehmen
# < 8k Schritte => andersrum
# sonst gleiche Warscheinlichkeit für 1kg oder -1kg
for _ in range(sims):
    w = []
    for i in range(days):
        lam = np.random.choice([5000, 15000], p=[0.6, 0.4], size=1)
        steps = np.random.poisson(lam)
        if steps > 10000:
            prob = [0.2, 0.8]
        elif steps < 8000:
            prob = [0.8, 0.2]
        else:
            prob = [0.5, 0.5]
        w.append(np.random.choice([1, -1], p=prob))
    outcomes.append(sum(w))

print("Probability of Weight Loss = {}".format(sum([x < 0 for x in outcomes])/len(outcomes)))




#### E-Commerce Ad Simulation
#
# Ad-Impression -> Click -> Signup -> Purchase -> Purchase-Value
#
# Anzahl der Ad Impression:
#    Poisson ZV mit lambda als normalverteilte ZV
# Klicken:
#   User hat binäre Entscheidung - klicken oder nicht-klicken
#   =>  Binomiale ZV - Rate of Success = Click-through-rate
#       benötigt histische Rate oder besser Verteilung der Rate
# Sign-up:
#   Binomiale ZV - analog zu Klicken mit Erfolgsrate = Sign-up-Rate
# Purchase:
#   Binomiale ZV - analog zu Klicken mit Erfolgsrate = Purchase-Rate
# Purchase-Value:
#   Exponentielle ZV: bestimmt durch historischen avg-purchase-value


## Sign-Up Simulation
# Ad-Impressions: als Poisson ZV mit
# λ normal verteilt mit mean = 100k besucher und sd = 2000
#
# Low-cost option:
# - CTR = 1%
# - Sign-up rate = 20%.
# Higher cost option:
# - kann CTR und SUR bis zu 20% erhöhen
#  => aber nicht sicher wieviel genau => also als uniform ZV
ct_rate = {'low':0.01, 'high':np.random.uniform(low=0.01, high=1.2*0.01)}
su_rate = {'low':0.2 , 'high':np.random.uniform(low=0.2,  high=1.2*0.2)}
def get_signups(cost, ct_rate, su_rate, sims):
    lam = np.random.normal(loc=100000, scale=2000, size=sims)
    # Simulieren von impressions(poisson), clicks(binomial) and signups(binomial)
    impressions = np.random.poisson(lam=lam)
    clicks = np.random.binomial(impressions, p=ct_rate[cost])
    signups = np.random.binomial(clicks, p=su_rate[cost])
    return signups

print("Simulated Signups = {}".format(get_signups('high', ct_rate, su_rate, 1)))

## Purchase Simulation
# Purchase Enscheidung: binimoal-verteilt mit rate Purchase-Rate = 10%
# Purchase Value: exponential verteilt mit mittlerem Purchase value = $1000
def get_revenue(signups):
    rev = []
    np.random.seed(0)
    for s in signups:
        # purchases als binomial, purchase_values als exponential
        purchases = np.random.binomial(s, p=0.1)
        purchase_values = np.random.exponential(scale=1000, size=purchases)

        rev.append(purchase_values.sum())
    return rev

print("Simulated Revenue = ${}".format(get_revenue(get_signups('low', ct_rate, su_rate, 1))[0]))

## Führt Ad-Redesign zu Verlust oder Gewinn
# Neuen Ad zu erstellen = 3000$
# Performance von neuen Ad = high version
sims, cost_diff = 10000, 3000

rev_low = get_revenue(get_signups('low', ct_rate, su_rate, sims))
rev_high = get_revenue(get_signups('high', ct_rate, su_rate, sims))

# calculate fraction of times rev_high - rev_low is less than cost_diff
x = [rev_high[i] - rev_low[i] < cost_diff for i in range(len(rev_low))]
fraction = sum(x)/len(rev_low)
print("Probability of losing money = {}".format(fraction))




#### Business-Modelling: Kostenoptimierung
# Beispiel: Getreideproduktion
# Ziel: Kostenoptimierung
# Getreide Produktion hängt ab von:
# - Regen
#   + kann nicht kontrolliert werden => Varianz
#   + normal verteilt mit mean=50, sd=15
# - Kosten=
#   + können kontrolliert werden
#   + erstmal fix: 5000
#
# Produziertes Getreide: Poisson ZV, wobei durschnittliche Produktion bestimmt durch:
#  100 × (cost)^0.1 × (rain)^0.2
cost = 5000
rain = np.random.normal(50, 15)

# Model für Getreide-Produktion
def corn_produced(rain, cost):
  mean_corn = 100 * (cost**0.1) * (rain**0.2)
  corn = np.random.poisson(mean_corn)
  return corn

# Simulieren der Getreide-Produktion
corn_result = corn_produced(rain, cost)
print("Simulated Corn Production = {}".format(corn_result))

def corn_demanded(price):
    mean_corn = 1000 - 8 * price
    corn = np.random.poisson(abs(mean_corn))
    return corn

## Berechnung für den Profit
def profits(cost):
    rain = np.random.normal(50, 15)
    price = np.random.normal(40, 10)
    supply = corn_produced(rain, cost)
    demand = corn_demanded(price)
    equil_short = supply <= demand
    if equil_short == True:
        tmp = supply*price - cost
        return tmp
    else:
        tmp2 = demand*price - cost
        return tmp2
result = profits(cost)
print("Simulated profit = {}".format(result))

## Kosten Optimierung
# Ziel ist es die Kosten zu bestimmen, welche den maximalen durchschnittlichen Gewinn geben
# => Kosten variieren und simulieren
sims, results = 1000, {}
cost_levels = np.arange(100, 5100, 100)

for cost in cost_levels:
    tmp_profits = []
    for i in range(sims):
        test = profits(cost)
        tmp_profits.append(test)
    results[cost] = np.mean(tmp_profits)

print("Average profit is maximized when cost = {}".format([x for x in results.keys() if results[x] == max(results.values())][0]))





## Monte-Carlo Integration
# 1. Gesamtfläche berechnen
# 2. Zufällig Punkte samplen
# 3. Multiplikation des Prozentsatzes der Punkte unter Kurve mit Gesamtfläche
def sim_integrate(func, xmin, xmax, sims):
    x = np.random.uniform(xmin, xmax, sims)
    y = np.random.uniform(min(min(func(x)), 0), max(func(x)), sims)
    area = (max(y) - min(y))*(xmax-xmin)
    result = area * sum(abs(y) < abs(func(x)))/sims
    return result

result = sim_integrate(func = lambda x: x*np.exp(x), xmin = 0, xmax = 1, sims = 50)
print("Simulated answer = {}, Actual Answer = 1".format(result))






#### Statistical Power
# Benötigte sample size durch simulation berechnen
import scipy.stats as st
np.random.seed(seed=0)

effect_size = 0.05
control_mean = 1
treatment_mean = control_mean * (1+effect_size)
control_sd = 0.5

sims = 1000
sample_size = 50
# sample size erhöhen um 10 bis man benötigte Power hat
while 1:
    control_time_spent = np.random.normal(loc=control_mean, scale=control_sd, size=(sample_size, sims))
    treatment_time_spent = np.random.normal(loc=treatment_mean, scale=control_sd, size=(sample_size, sims))
    # treatment_time_spent = control_time_spent * (1+effect_size) GEHT NICHT!!
    # => nicht ähnlicher Ausgang - braucht nur ca. häfte => siehe nächste Methode

    t, p = st.ttest_ind(treatment_time_spent, control_time_spent)

    # Power ist der Prozentsatz der Simulationen in denen der P-Wert kleiner als 0.05
    power = (p < 0.05).sum()/sims
    if power >= 0.8:
        break
    else:
        sample_size += 10
print("For 80% power, sample size required = {}".format(sample_size))

control_time_spent.shape
select_index = np.random.choice(control_time_spent.shape[0], size=100, replace=False)
sample = control_time_spent[select_index]
sample.shape



#### Power Berechnung aus derselben Grundgesamtheit + shiftet
sims = 1000
sample_size = 50
# Verteilung auf der man den Effekt nachher feststellen soll
control_time_spent_full = np.random.normal(loc=control_mean, scale=control_sd, size=(10000, sims))
#control_time_spent_full = np.random.normal(loc=control_mean, scale=control_sd, size=(100, sims))
# => bei kleiner ausgangsmenge muss bootstrapping mit replace = True gemacht werden
#    ==> liefert sehr ähnliches Ergebnis
### SHIFT - treatment effekt auf der population nachstellen!!!
treatment_time_spent_full = control_time_spent_full * (1+effect_size)

while 1:
    select_index_control = np.random.choice(control_time_spent_full.shape[0], size=sample_size, replace=True)
    control_time_spent = control_time_spent_full[select_index_control]
    select_index_treatment = np.random.choice(treatment_time_spent_full.shape[0], size=sample_size, replace=True)
    treatment_time_spent = treatment_time_spent_full[select_index_treatment]
    #  => jedes mal neues sample generieren!!!

    t, p = st.ttest_ind(treatment_time_spent, control_time_spent)
    power = (p < 0.05).sum()/sims
    if power >= 0.8:
        break
    else:
        sample_size += 10
print("For 80% power, sample size required = {}".format(sample_size))





#### Bootstrapping
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


#### Jackknife
heights = np.random.normal(size=1000,loc=179,scale=20)
mean_lengths, n = [], len(heights)
index = np.arange(n)

for i in range(n):
    jk_sample = heights[index != i]
    mean_lengths.append(np.mean(jk_sample))

mean_lengths = np.array(mean_lengths)
print("Jackknife estimate of the mean = {}".format(np.mean(mean_lengths)))

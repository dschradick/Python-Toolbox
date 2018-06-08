########## VISUALISIERUNG - MATPLOTLIB
# Figure = äußserster container, welcher für matploblib grafik
# => kann mehrere Achsen enthalten
# Axes = eim plot oder graph

## !!! REF-BUCH: Python for Data Science Handboook!! ##
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from bokeh.sampledata.autompg import autompg as auto
from bokeh.sampledata.iris import flowers as iris
import string
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

#### Test-Daten erzeugen
a = [x for x in range(10)]
b = [x * 1.5 for x in a]
c = [x ** 2 for x in a]
mu = 100; sigma = 15; x = mu + sigma * np.random.randn(0)


#### BASICS

### Einfacher Plot
plt.plot(a,b,'o')     # Punkte
plt.plot(a,b,'-')     # Linie
plt.plot(a,b,'go--')  # grün + Punkte + gesrichtelte Linie

### Vollständiger Plot: Titel, Legende, Ticks,...
plt.title('Testplot'.upper())
plt.xlabel('Y Achse (%)')
plt.ylabel('Y Achse ($)')
plt.grid(True,which='major')
plt.text(2, 2, r'$\mu=100,\ \sigma=15$')
plt.xticks(range(11),string.ascii_lowercase[:11],rotation=45)
plt.plot(a,b,'o')
# Prozent Labels
fig, axes = plt.subplots()
fmt = '%.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)



### Plot löschen
plt.clf()
plt.figure() # fängt neue Figure an

### Subplots
# subplot(nrows,ncols,nsubplot) -
# => Reihenweise, oben links anfangend, index startet mit 1
plt.clf()
fig, axes = plt.subplots(1, 2, figsize=(12,5))
plot(ax=axes[0])
#
plt.subplot(2,1,1); plt.plot(a,b,'green')
plt.subplot(2,1,2); plt.plot(a,c,'red')
plt.tight_layout(); plt.show()

### In einem Plot: Mehrere Graphen
# -- = dashed, =. = dashed-dotted, s = rechtecke, ^ = Dreiecke
plt.plot(a, a, 'y-.', a, b, 'bs--')
# Alternativ:
plt.plot(a,b,'green') ; plt.plot(a,c,'red') ; plt.show()


### Plot speichern
plt.savefig('plot.png')


### Legende + Annotatons + Arrows
plt.clf()
plt.plot(a,b,'green',label='y = x * 1.5')
plt.plot(a,c,'red',label='y = x^2')
#plt.annotate('x^2',xy=(2,12)) # einfache annotion
plt.annotate('x^2',xy=(4,16),xytext=(1,30),arrowprops={'color':'red'}) # annotation mit pfeil
plt.legend(loc='best') # oder: "upper right"
plt.show()


### Pie-Chart
_ = plt.pie([20,10,40],labels=['Teil1','Teil2','Teil3'],autopct='%1.1f%%')
plt.show()


### Histogram einfach
x = np.random.normal(0.5, 0.3, 1000)
plt.clf()
bins = [0,70,100,130,200]                          # häufig besser: explizite bin grenzen
counts,bins_,patches = plt.hist(x,bins=20)         # anzahl von bins festlegen
counts,bins_,patches = plt.hist(x,bins=bins)       # explizite bin grenzen
counts,bins,patches = plt.hist(x,histtype='step')  # smooth und unausgefüllt
plt.show()

### Density
# mit seaborn
import seaborn as sns
data = np.random.exponential(3, 10000)
sns.distplot(data)

### Überlagert: Histogram & Density 
fig, ax = plt.subplots()
n, bins, patches = ax.hist(x, 50, normed=1)
y = mlab.normpdf(bins, 0.5, 0.3)
ax.plot(bins, y, '--')
ax.set_xlabel('IQ')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
fig.tight_layout()
plt.show()


### Styles
plt.style.available
plt.style.use('seaborn') # oder z.B. ggplot


#### 2-D Arrays & bivarite Funktionen
import numpy as np
#u = np.linspace(-2,2,3) # erzeugt 1D-Array mit Werten mit gleichem Abstand (von,bis,AnzDatenpunkte)
#v = np.linspace(-1,1,5)
u = np.linspace(-2,2,6) # Zeigt Funktion besser
v = np.linspace(-1,1,10)
# Meshgrid erzeugt ein zwei 2D-Array mit gleicher Shape mit den Werten Konstant entlang jeweils verschiedener Achse
# die paare entsprechen den punkten auf einem rechteckigen grid
# => erlaubt es bivarite funktion im grid zu untersuchen / plotten
# Y,X = np.meshgrid(range(10),range(20)) # => erzeugt (20,10) 2-D Array
X,Y = np.meshgrid(u,v)
(X[0,0],Y[0,0])
point = (lambda x,y: (X[0,x],Y[y,0]))
point(0,0)

## Zu Plotten: 2D-Funktion (hier: Gradient)
# f(X,Y) = Z
Z = X**2/25 + Y # bivariate Funktion Z - durch numpy-vektor-opertion (kann aber anders sein)

# Plotting (des 2D-Arrays)
plt.subplot(2,2,1)
plt.set_cmap('gray') # autumn
plt.pcolor(X,Y,Z) # X,Y angeben für Mesh-Grid Achsenbeschriftung
plt.colorbar()
#### oder...
plt.subplot(2,2,2)
plt.colorbar()
plt.pcolor(X,Y,Z,cmap='viridis')
plt.subplot(2,2,3)
plt.contour(X,Y,Z,30,cmap='autumn')
plt.colorbar()
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,30,cmap='winter')
plt.colorbar()
plt.axis('tight')
plt.show()

auto.head(2)

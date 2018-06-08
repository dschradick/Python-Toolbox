########## BASICS
# ALT+i => hydrogen-inspector (z.B. Funktions-Definition)
import numpy as np
import pandas as pd


### AUSGABE
# Formatieren
# float    = {positon:stellen_insgesamt.stellen_nachkommaf}
# int      = {positon:stellen_insgesamtd}
# prozent  = {positon:stellen_insgesamt.stellen_nachkomma%}
# => stelleningesamt kann weggelassen werden
# {var_name} statt {0} => postion kann benannt werden
np.set_printoptions(precision=3)
x = 122.345; y = 67.890; z = 10
f = 10.2345; i = 10;  p=0.15
"The number is {}".format(5)
"The number is {number}".format(number=i)
"The number is {0:9.4f}".format(f)
"The number is {0:d}".format(i)
"The number is {0:10.1%}".format(p)
# Tabelle
for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))


### DIVERSES
## Hilfe
?pd.read_csv # in iphython shell
## Testen
assert 1 == 1
## Kein Zeilenumbruch bei print()
print("Daniel" + " ",end="")
##  IPython Magic
# %lsmagic - alle magics
# ! führt Befehl aus, %cd newdir
# %%time
!ls
x = !ls

# Funktionen und variablen im modul
dir(a)


## Boolean
True == 1
False == 0
x = [True,True,False]
sum(x) # => 2

#### ?!
condition = True
1 if condition else 0

#### COMPREHENSION
## List Comprehesion
[x * 2 for x in range(10)]
## Dictionary Comprehension
names = {'Joana', 'Daniel','Corbinian', 'Justine'}
dictonary = {name:len(name) for name in names}


## Splat-operator: flatten list
a = [1,2,3,[1,2]]
print(a)   # (1, 2, 3, (1, 2))
print(*a)  # 1 2 3 (1, 2)

## Kein Output
_ = 3 + 3
print(_)
3 + 3;


## Substring
my_string = 'Daniel'
my_string[0:2]    # Ersten zwei: 'Da'
my_string[-2:]    # Letzen zwei: 'el'
my_string[::-1]   # Reverse: 'leinaD'


## Datei Handling
# with = context-manager => kein close() notwendig
# alternativ file = open("test.txt",mode="r") => file.close() notwendig!
# "rb" / "wb" für binary
filename = "/tmp/test.txt"
with open(filename,mode="w") as file:
    file.write("Hallo\nDaniel!")
with open(filename,mode="rb") as file:  # rb für binary
    line = file.readline()
    content = file.read()
with open(filename,mode="r") as file:
    for line in file:
        print(line)



## Beispiel-Daten erzeugen
list(range(10))
list(range(5,10))
x = np.linspace(-1, 1, 10) # erzeugt 1D-Array mit Werten mit gleichem Abstand (von,bis,AnzDatenpunkte)
f_x2 = (lambda x: x ** 2)
y = [f_x2(x) for i in x]
y
## Regular Expressions
import re
pattern = re.compile('^\$\d*\.\d{2}$')
result = bool(pattern.match('$17.89')) # match() liefert match-obj (oder None)

## Funktionen
# *args wenn man die anzahl der args nicht vorher festlegen will
# def count_entries(df, *args):
#   for col_name in args:
g_var = 5
def raise_to(value1, value2=2,**kwargs):
    """Raise value1 to the power of value 2 and vice verca"""
    global g_var  # verwende global variable g_var

    # Just for fun
    for key, value in kwargs.items():
        print(key + ": " + value)

    new_value1 = value1 ** value2
    new_value2 = value2 ** value1
    new_tuple = (new_value1, new_value2) #immutable (vs list)
    return new_tuple



### LISTEN:
# => lists, iterators, enumartors, generators

## List
a = ['a','b','c']       # list
x1, x2, x3 = a          # => unpacking
a_list = list(range(10))
x = [1,2,3]
for a,b in zip(x,x):
    print(a,b)        # 1,1 ; 2,2 ; 3,3

## Iterators
# Erzeugen
i = iter(a)
# Zugriff
next_object = next(i)
for x in i:
    print(x,end="")

## Enumerator
# => erzeugt index,value in for loop
for index,value in enumerate(a,start=1) :
    print(index,value)

# Looping in dict
for key in {'one':1, 'two':2}:
    print(key)

#### Python data structures & collections
a = [1,2,3]
[0] + a; a.append(4); a.remove(4); a.extend([4])
a.sort();
a.count(4)
sorted(a)
list(reversed(a))
# normale dictionaries und sets benutzen hash-tables
d = {'a':'1',b:'2'} # => benutzt hash-tables
s = set(['a','b']); s.add('hallo'); s.add('welt'); s
'a' in s; 'a' not in s
s - s; s & s # intersection
from collections import Counter, defaultdict, OrderedDict, deque
od = OrderedDict() # Ordnung gemäß Einfügung
od['test'] = 2; od['ab'] = '1';
d = dict(); d['hallo'] +=  1                 # Fehler
dd = defaultdict(int); dd['h'] = dd['h'] + 1 # Funktioniert
cnt = Counter({1:3,2:4})
list(cnt.elements())          # [1, 1, 1, 2, 2, 2, 2]
cnt.most_common()             # [(2, 4), (1, 3)]
cnt['a'] += 10

# Looping mit looping index
a = ['a','b','c']
for x,i in enumerate(a):
    print(i)
# looping über mehrere listen gleichzeitig
a = [1,2,3]; b = [4,5,6]
for aa, bb in zip(a, b):
    print('{0} : {1}'.format(aa, bb))

## Dictonary
key_list = [1,2,3]; value_list = ['a','b','c']
zipped_lists = zip(key_list, value_list)
rs_dict = dict(zipped_lists)

## List comprehension
# [output expression for iterator variable in iterable if predicate expression]
#   => geht für alle iterators
squares = [x**2 for x in range(10)]
# bedingung an iterable
squares_of_two = [x**2 for x in range(0,10) if x % 2 == 0]
# bedingung an Output von iterable
squares2 = [x**2 if x % 2 == 0 else 0 for x in range(0,10)]
# veschachteln => erzeugt 5x5 matrix mit jeweils 1-5
matrix = [[col for col in range(0,5)] for row in range(0,5)]
# tupel / tabelle mit zwei spalten und 10 Einträgen
two_tupel = [(num1,num2) for num1 in range(1,5) for num2 in range(1,5)]
# für daten-transformation
names = {'Daniel', 'Joana,','Corbinian', 'Justine'}
initals_with_middle_an = [name[0] for name in names if name[1:3] == 'an']

## GENERATORS
# = lazy evaluation (können als paramter (für grosse datenmengen) übergeben werden)
squares_generator = (x**2 for x in range(10)) # Generator Expression mit () statt [] sonst wie list comprehension
def num_sequence(n):                          #  Generator Function (yield statt return)
    i = 0
    while i < n:
        yield i
        i += 1

## Zip
a = [1,2,3]
zip(a,a)                       # iterator
list(zip(a,a))                 # zur liste
r1,r2 = zip(*list(zip(a,a)))   # = a,a
for value_1, value_2 in zip(a,a):
    print(value_1,value_2)
# 1 1
# 2 2
# 3 3


## Lambda Funktionen
make_binary_category = lambda x: 0 if x < 5 else 1
raise_to_power = lambda x, y: x ** y
y = raise_to_power(2,3)  # 8


#### MAP, FILTER, REDUCE
from functools import reduce
nums = [1,2,3,4]
square = lambda x: x**2
even = lambda x: x % 2 == True
add = lambda x,y: x + y
# map() wendet eine Funktion auf ein Objekt wie ein Liste an
square_array = list(map(square,nums))
# filter() filtert Elemente aus Liste, welche NICHT(!) dem Kriterium entsprechen
odd_array = list(filter(even,nums))
# reduce() liefert einzigen Wert als Resultat einer Operation (mit 2 Params) eine Liste
sum_of_list = reduce(add,nums)
sum_of_list
# => kann alles auf dataframe angewendet werden



#### EXCEPTIONS
# => raise / try-except
def sqrt(x):
    if (x < 0):
        raise ValueError("x must be non-negative")
    try:
        return (x ** 0.5)
    except TypeError:
        print ("x must be an int")



# SLICING
# [start:stop:stride]
# => start - inclusive; stop - exclusive
# -x bei pos     = vom Ende ausgehend ;
# -x bei stride  = rückwärtsgehen
a = [0,1,2,3,4,5,6,7,8,9]
b = [10,11]

a[0:8]       # 0 bis exclusive 8
a[0:8:2]     # 0 bis exclusive 8 in zweier Schritten
a.index(2)   # => liefert den index-array
del(a[:3])   # löschen von Elementen INPLACE


c = a + b    # !! hintereinanderhänge => vs numpy vektor addition
del(c[-3:])
c


#### Numpy
# Vektoroperationen
import numpy as np

## Daten importieren
# besser: pandas
#np.loadtxt("data.txt",delimiter=',',skiprows=1,dtype='str')

## Numpy array aus list
a_list = [0,1,2,3,4,5,6,7,8,9,100]
b_list = [0,1,2,3,4,5,6,7,8,9,200]
a = np.array(a_list)
b = np.array(b_list)

## Erzeugen von Daten
x1 = np.linspace(1, 2, num=4)  # [1, 1.33, 1,66, 2]
x2 = np.arange(1,5)            # [1, 2, 3, 4]
x3 = np.logspace(1,100,3)
x3


## Vektor-Operationen
c = a + b            # Elememtweise Addition
c_np = c * 2         # Elementweise Mulitplikation mit Skalar

## Auswahl
a_short = a[1:5]     # Erste Spalte

## Suchen und sortieren
#https://docs.scipy.org/doc/numpy/reference/routines.sort.html
a = a[::-1]          # reverse
a.sort()             # sortieren

# Suchen
np.amax(a)           # größtes element
np.argmax(a)         # index mit größtem element
np.argwhere(a > 5)   # liefert indizes, dessen elemente entsprechen
# = a > 5

## Boolean Indexing  (zur Auswahl von Reihen)
a_index = a > 5           # Liefert Index - Liste gleicher länge mit True/False
a_subset = a[a > 5]       # Anwendung des Index zum Subsetting
a_subset_2 = a[a_index]   # identisch

## Datentypen
a = a.astype(float)
a = a.astype(np.int32)
a.dtype

## Zufallszahlen
np.random.seed(0)
np.random.rand()                         # Zufallszahl [0,1)
np.random.randint(10)                    # Zufallszahl 0 bis exclusive 10
np.random.choice(range(0,10),size=2)     # Zufällige Auswahl


## Einfache Statistiken
a.mean()
a.std()
a.var()
np.median(a)
np.percentile(a,50)
np.percentile(iris.petal_length,[25,50,75])
np.corrcoef(a,b)[0,1]


## Verteilungen
# Zufallszahlen: .rvs(loc=0, scale=1, size=1, random_state=None)
# PDF:           .pdf(x, loc=0, scale=1)
# CDF:           .cdf(x, loc=0, scale=1)
# Inverse CDF    .ppf()
# Random Sampling
import scipy.stats as stats
stats.norm.rvs(loc=50, scale=5, size=2, random_state=0)

# Zufallszahlen mit numpy
np.random.uniform(10, size=100)
np.random.normal(loc=0, scale=2, size=100)
np.random.poisson(lam=3, size=100)


#### Logging
import logging
logging.debug('Debugging information')
logging.info('Informational message')
logging.warning('Warning:config file %s not found', 'server.conf')
logging.error('Error occurred')
logging.critical('Critical error -- shutting down')



## Plots
import matplotlib.pyplot as plt
plt.scatter(x = a, y = b, s = b, c = b, alpha=0.8)

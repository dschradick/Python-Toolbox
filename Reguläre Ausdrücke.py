########## REGULÄRE AUSDRÜCKE
# Müssen als positives match ausdedrückt werden
# r"foo" für raw => einfache blackslashes
import re

####!!!!!!####!!!!!####
# GUTE ÜBERSICHT in:  PYTHON Pocket Reference

### Match-Objekt Methoden
# group()	Return the string matched by the RE
# start()	Return the starting position of the match
# end()	    Return the ending position of the match
# span()	Return a tuple containing the (start, end) positions of the match

### Zeichenauswahl
# [ab]  = Zeichen aus Auswahl (hier a oder b)
# [^ab] = ausser angegebene Zeichen - Komplement (hier alls ausser a oder b)
#   => ^ in [] also andere Bedeutung!!!

# Matchgroups 
re.match(r"(a)(b)\1\2", "abab" ) # √
# Names matchgroup => gut für parsing 
r'(?P<id>[a-zA-Z_]\w*)' #
r'a(?=x)' # positive lookahead assertion => x must follow but is not part of the match - 
          # matches "ab", but only "a" is returned as match

### Zeichenklassen (gross = negation)
# \d = Zahl = [0-9]
# => Negation: \D = keine Zahl = [^0-9] = []
# \w = Buchstabe,Ziffer,Unterstrich = [a-zA-Z0-9_]
# \s = Leerzeichen
# \. = .

### Spezielle Zeichen
# ^ = Stringanfang
# $ = Stringende
# [^0-9] = Komplement (hier: keine zahl)

+#### Quantoren
# Der voranstehende Ausdruck...
# {n} = genau n-mal = {n,n}
# {n,m} = mindestens n-mal, maximal m-mal
# ? = optional = null oder einmal = {0,1} 
# + = mindestens einmal {1,}
# * = darf beliebig oft {0,}

## Non greedy Quantoren
# Non-greedy quantifiers match the same possibilities as their corresponding normal (greedy) counterparts, 
# but prefer the smallest number rather than the largest number of matches.
# *?	non-greedy version of *
# +?	non-greedy version of +
# ??	non-greedy version of ?
# {m}?	non-greedy version of {m}
# {m,}?	non-greedy version of {m,}
# {m,n}?	non-greedy version of {m,n}

#### ASSERTIONS: Lookahead & Lookbehind 
# https://www.rexegg.com/regex-disambiguation.html#lookarounds

# (?=<lookahead_regex>) 
# Creates a positive lookahead assertion (but not itself part of match)
#   = captured match must be followed by whatever is within the parentheses but that part isn't(!!) captured.
#   = asserts that what follows the regex parser’s current position must match <lookahead_regex>:
# eg.
# '\d+(?=\.\w+$)'
# file4.txt will match 4.
# file123.txt will match 123.

# ^	matches at the beginning of the string
# $	matches at the end of the string

# (?=re)	positive lookahead matches at any point where a substring matching re begins (AREs only)
# (?!re)	negative lookahead matches at any point where no substring matching re begins (AREs only)
# (?<=re)	positive lookbehind matches at any point where a substring matching re ends (AREs only)
# (?<!re)	negative lookbehind matches at any point where no substring matching re ends (AREs only)

r'a(?=x)' # positive lookahead assertion => x must follow but is not part of the match - 
          # matches "ab", but only "a" is returned as match

## Lookarounds KONSUMIEREN NICHT!!!!
# \d+(?= Euro)  => Matches: "100 Euro" => but returns "100" as match

## Bsp
# .*([aeiou]).*\1.*  # Enthält zweimal denselben vokal
# ^(?!Test).*$       # fängt nicht mit Test an
# ^((?!Test).)*$     # enthält nicht Test (an jeder Position)




#### Pandas
mtcars.name.str.match(r"(^\w*)")     # liefert binary mask
mtcars.name.str.extract(r"(^\w*)")   # zum erzeugen neuer Spalte
mtcars.replace(..., regex=True)
# => alternativ auch mit apply(func) & dann in func ausführlich re package verwenden
df.mileage.str.extract("(\d+.\d+.*(?=km/kg))|(\d+.\d+.*(?=kmpl))", expand=True) 
# ?= look ahead assertion => asserted to follow but not part of match
# zwei spalte aus "26.6 km/kg" & "19.67 kmpl" => zwei verschiene typen von einträgen in einer Spalte 

#### Kompilieren
# Reguläre Expression durch kompilieren zu Pattern-Obj für Operationen wie suchen/ersetzen
# (r für raw -> kein problem mit backslashes)
p = re.compile(r'ab+',re.IGNORECASE)


#### Matching
## Matching des Anfangs des Strings - prüft Anfang(!) des Strings
match = p.match('abbbccc') # kein match => None

## Finden des Pattern an beliebiger Stelle im String mit search
p.search('babbbccc')
print(match)
if match:
    print(match.group()) # => abbb
else:
    print("No Match")

## Alle matches finden
for match in p.finditer('abbbcabcc'):
    print("Match: ",match.group())

## Splitting
p.split('cabbbd', maxsplit=0) # ['c','d']

## Ersetzen
# () für matchgroup (kann auch schon im ersten string für wiederholung benutzt werden)
re.sub(r'(\+49)(800|\d{4})(\d+)',r'\1 \2 \3','+494131423')  # +49 4131 423


#orders.customer_name.str.extract(r"(.*)\s *.") # vorname
#"((?:19|20)\\d{2})(?!.*\\d{4})"
#extract(title, "year", "(20\\d{2})", convert = TRUE, remove = FALSE) # four digits i a row; 
# () für match-group
# (?:19|20) non-matching group

#### FÜR LOG-PARSING => DICTONARY 
## Aus komplexen match mit Variablen nach Dictonary
format_pat= re.compile(
    r"(?P<host>[\d\.]+)\s"
    r"(?P<identity>\S*)\s"
    r"(?P<user>\S*)\s"
    r"\[(?P<time>.*?)\]\s"
    r'"(?P<request>.*?)"\s'
    r"(?P<status>\d+)\s"
    r"(?P<bytes>\S*)\s"
    r'"(?P<referer>.*?)"\s'
    r'"(?P<user_agent>.*?)"\s*')   # => wird zu einem String
logPath = "~/Documents/Data/access_log.txt"

URLCounts = {}
with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            split = request.split()
            if len(split) < 3:
                print(split)
                continue
            (action, URL, protocol) = split
            if URL in URLCounts:
                URLCounts[URL] = URLCounts[URL] + 1
            else:
                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))

s = "Hallo Welt"

#### Casting
i = 10
s = str(i)
i = int("10")

#### Zugriff & Looping
s[0]; s[-1]                        # Anfang und Ende
s[0:len(s)]                        # von-bis
s[::-1]                            # reverse
for c in s:
    print(c,end='')


#### Inhalt scannen
'allo' in s                        # contains
s.startswith('Hallo')
s.endswith('Welt')
s.find('Welt')                     # Index


#### Säubern
s.replace("Welt",'Erde')
s.strip()    # vorder & hintere Leerzeichen entfernen
s.rstrip()
s.lstrip()


#### String <=> Array
array = 'Wörter_durch_Strich_getrennt'.split()
"_".join(['Wörter','durch','Strich','getrennt'])


# Manipulation
s + " " + s
s.upper()
s.lower()
s.capitalize()

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

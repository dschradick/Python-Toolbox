########## PYTEST
## Grundsätze:
# Wichtige Fälle: 0,1 und n
# Jeder Test: One Reason to fail
# Tests auf einer Abstraktionseben höher als der geschriebene Code

## TDD
# Ziel ist weniger testen an sich, sondern Design / Spezifikation
# 1. Fail-test schreiben (und testen)
# 2. Code schreiben der Test erfüllt
# => KEINEN CODE schreiben für den es keinen Test gibt / welcher nicht zur Test-Erfüllung beiträgt

## VS Unitest
# Unittest ist der Aufbau durch Klassen wie in java (mit Fixture-methoden,setup,teardown)und assertequals)
# Pytest führt auch automatisch Tests von Unittest aus!
# => kann kombiniert werden (bis auf fixtures)
# => nicht so pythonic



import pytest
# => nur für raises, Pytest braucht keine imports

#### Tests ausführen
# Dateiname muss mit "test_" beginnen
# Mit "python -m pytest" im Verzeichnis ausführen

## Test Coverage
# python -m pytest --cov-report term-missing --cov
# python -m pytest --cov-report html --cov
#  => erzeugt htmlcov verzeichnis mit HTML-Report
# Coverage = Prozent der zeilen die beim test ausgeführt worden sind
# Missing  = welche Zeilen wurde nicht getestet
# => wenn Funktion nicht getestet werden soll nach Funktion-deklaration: # pragma: no cover

class Phonebook():

    def __init__(self):
        self.entries = {}

    def add(self,name,number):
        self.entries[name] = number

    def lookup(self,name):
        return self.entries[name]

    def names(self):
        return self.entries.keys()

    def numbers(self):
        return self.entries.values()

    def clear(self):
        pass # z.B. Datei schliessen

    def not_tested(self):
        pass

## Einfachster Fall
def test_add_and_lookup_entry():
    phonebook = Phonebook()
    phonebook.add('Bob','123')
    assert "123" == phonebook.lookup("Bob")


# Analog zu setup bei unittest
# aber hier durch sowas wie dependency injection
@pytest.fixture
def phonebook():
    return Phonebook()

# Fixture mit Teardown
# (Fixture mit temporären Verzeichnis - def phonebook(tmpdir):)
@pytest.fixture
def phonebook(request):
    phonebook = Phonebook()
    def cleanup_phonebook():
        phonebook.clear()
    request.addfinalizer(cleanup_phonebook)
    return phonebook

def test_phonebook_gives_access_to_names_and_numbers(phonebook):
    #pytest.skip('WIP') # wird im Output angezeigt
    phonebook.add('Alice',"12345")
    phonebook.add('Bob',"123")
    assert(phonebook.names() == {'Alice','Bob'})
    assert "123" in phonebook.numbers()

def test_missing_entry_raises_KeyError(phonebook):
    with pytest.raises(KeyError):
        phonebook.lookup("missins")

def add(a,b):
    return a + b

# Parametrisierter Tests
examples = (('value_1', 'value_2', 'the_sum','comment'), # comment für debugging
            [(0,0,0,'adding zero'),
             (1,1,2,'adding one'),
             (-1,2,1,'adding negative')])
@pytest.mark.parametrize(*examples)
def test_adding(value_1,value_2,the_sum,comment):
    assert the_sum == add(value_1,value_2)

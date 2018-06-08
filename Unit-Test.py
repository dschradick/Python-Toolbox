import unittest
# Befehl: nosetests Unit-Test.py

## Verschiedene Asserts
# assert()
# assertEqual(a, b)
# assertNotEqual(a, b)
# assertIn(a, b)
# assertNotIn(a, b)
# assertFalse(a)
# assertTrue(a)
# assertIsInstance(a, TYPE)
# assertRaises(ERROR, a, args)


## first
#class Calculator():
#    def add(self, a, b):
#        pass
class Calculator():
    def add(self, a, b):
        return a + b

class CalculatorTest(unittest.TestCase):
    def test_calculator_add_method(self):
        calc = Calculator()
        result = calc.add(2,2)
        self.assertEqual(4, result)



if __name__ == '__main__':
    unittest.main()

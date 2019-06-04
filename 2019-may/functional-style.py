#functions are first class objects i.e., they can be used like other data objects
funcs = [sum, len, type]
for func in funcs:
    print(func(range(1,5)))

#anonymous functions: functions without name
funcs = [lambda x: x, lambda x: x**2, lambda x: x**3]
for func in funcs:
    print(func(10))

#higher order functions: functions that accepts function as argument
def f(a, b, g):
    return g(a, b)
f(1, 2, lambda x, y: x + y)
f(1, 2, lambda x, y: x * y)

#functions returning other functions
def f(a):
    def g(x):
        return a + x
    return g
g = f(3)
g(5)

#some built-in higher order functions in python
x = map(lambda x: x**2, range(4))
print(type(x))
print(list(x))

def square(x):
    return x*x
y = list( map(square, range(1, 10)) ) 
print(y)

z = filter(lambda x: x % 2 == 0, range(4))
print(list(z))

from functools import reduce
import numpy as np

reduce(lambda x, y: x + y, range(1, 5) )
reduce(lambda x, y: x + y, range(1, 10) )
reduce(lambda xs, ys: xs + ys, [[1,2], [3,4], [5,6]])
reduce(lambda xs, ys: xs + ys, [np.array([1,2]), np.array([3,4]), np.array([5,6])])

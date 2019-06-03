#logic is divided as a group of functions

#type conversion
'abc' + str(10)
10.4 + int(20.6)

#check the type of object at runtime
a = 10
print(type(a) == int)

#custom function
def add(a, b, c):
    return a+b+c

print(add(10, 20, 30))
print(add(10.4, 'abc', 30))
print(add('a', 'abc', 'c'))


#custom function
def add(a, b=10, c=20):
    return a-(b+c)

print(add(30,10,20))
#skip b , pass a and c
print(add(30, c = 20))
print(add(a = 30, c = 20))
print(add(c = 20, a = 30))

#return multiple values from a functin
def f(a, b):
    return a+b, a-b

c, d = f(10,20)

import numpy as np

def g(n1, n2):
    return np.ones((n1,)), np.zeros((n2,))

a1, a2 = g(3, 5)

#how to create variable argument functions
def add1(*args):
    print(args)
    res = 0
    for x in args:
        res = res + x
    return res

input1 = [1, 2, 3, 4, 5]
print(add1(*input1))

input2 = list(range(1, 5))
print(add1(*input2))

def add2(**kargs):
    print(kargs)
    res = 0
    for x in kargs.values():
        res = res + x
    return res

kinput1 = { 'x1': 10, 'x2':20.5, 'x3':34} 
print(add2(**kinput1))

#why do we need variable arguments?




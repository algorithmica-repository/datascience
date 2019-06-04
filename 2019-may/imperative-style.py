import numpy as np
import time

#logic is divided as a group of functions

#type conversion
'abc' + str(10)
10.4 + int(20.6)

#custom function
def add(a, b, c):
    return a+b+c

print(add(10, 20, 30))
print(add(10.4, 'abc', 30))
print(add('a', 'abc', 'c'))


#custom function with default arguments
def add(a, b=10, c=20):
    return a-(b+c)

print(add(30,10,20))
#skip b , pass a and c
print(add(30, c = 20))
print(add(a = 30, c = 20))
print(add(c = 20, a = 30))

#function with mutable default arguments
def h1(a, b=[]):
    b.append(a)
    return a, b
print(h1(1))
print(h1(2))

def h2(a, b=None):
    if b == None:
        b = []
    b.append(a)
    return a, b
print(h2(1))
print(h2(2))

#return multiple values from a function
def f(a, b):
    return a+b, a-b
c, d = f(10,20)

def g(n1, n2):
    return np.ones((n1,)), np.zeros((n2,))
a1, a2 = g(3, 5)

#why do we need variable arguments? when we donot know the number of arguments in advance
#functions with variable number of arguments
def sum1(*args):
    print(args)
    res = 0
    for x in args:
        res = res + x
    return res
print(sum1(1, 2))
print(sum1(1, 2, 3))
print(sum1(*[1, 2, 3]))

def sum2(**kargs):
    print(kargs)
    res = 0
    for x in kargs.values():
        res = res + x
    return res
print(sum2(x1 = 10, x2 = 20.5) ) 
print(sum2(x1 = 10, x2 = 20.5, x3 = 22) ) 
print(sum2(**{ 'x1': 10, 'x2':20.5, 'x3':34} ))

def sum3(*args, **kargs):
    print(kargs)
    res = 0
    for x in args:
        res = res + x
    for x in kargs.values():
        res = res + x
    return res
print(sum3(10, 20, 30, x1 = 10, x2 = 20.5) )
 
#usage of variable arguments: time decorator
def g(f, *args, **kwargs):
    start = time.time()
    ans = f(*args, **kwargs)
    elapsed = time.time() - start
    return(ans, 'Function took %s seconds' % elapsed)

res = g(sum1,1,2,3)
print(res)
res = g(sum2, x=10, y=20)
print(res)
res = g(sum3, 5, 10, x=10, y=20)
print(res)

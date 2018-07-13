def add1(a, b):
    return a+b 

print(add1(10, 20))

def add2(a, b, c=10):
    tmp = a + b
    return tmp + c

print(add2(10, 20, 30))
print(add2(10, 20))
print(add2(b=10, c=20, a=20))

def map(f, a) :
    return f(a[0], a[1])

map(add1, [10, 20])

#logic is divided into bunch of routines
def add2(a,b):
    return a+b

print(add2(10,20))
print(add2(10.1,20.4))
print(add2('abc','def'))

def add3(a,b,c=10):
    return a+b+c
print(add3(1,2,3))
print(add3(1,2))

def sub3(a,b=5,c=10):
    return a-b-c
#pass a&b without c
print(sub3(10,20)) 

#pass a without b & c
print(sub3(10))  

#pass a&c without b
print(sub3(a=10,c=5))

print(sub3(c=5,a=10))

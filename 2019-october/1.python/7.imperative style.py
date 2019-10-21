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
print(add(30, c=20))
print(add(a = 30, c = 20))
print(add(c = 20, a = 30))
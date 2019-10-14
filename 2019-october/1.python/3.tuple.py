t1 = (10, 20, 'abc', True, 30.6)
print(type(t1))
print(t1)
print(id(t1))

t2 = t1
print(t2)
print(id(t2))

print(t1 == t2)

t3 = (10, t1, 'abc', True)
print(t3)
print(type(t3))

t1[0] = 20

#element access
t1[2]
t1[1:4]
t1[1:4:2]
t1[2:]
t1[:3]

#enumeration over list
for x in t1:
    print(x)

for i,x in enumerate(t1):
    print(i, x)

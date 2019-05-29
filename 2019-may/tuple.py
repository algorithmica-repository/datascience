#tuple: An immutable indexed container of data
t1 = (10, 20, 30.5, True, 'abc')
print(t1)
print(type(t1))

print(t1[1])
print(t1[:3])
print(t1[-1])

t1[0] = 60

t2 = (100, t1)
print(t2)

for x in t1:
    print(x)
    
for i,x in enumerate(t1):
    print(i, x)

t1 = (10, 20, 30, 40)
print(type(t1))
print(t1)


t2 = (10, 20.5, False, 'abc')
print(type(t2))
print(t2)

#get the tuple elements
print(t2[2])
print(t2[0:3])
print(t2[1:])
print(t2[:3])

#loops on list
for x in t1:
    print(x)
for i,x in enumerate(t1):
    print(i,x)


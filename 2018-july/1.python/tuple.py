tuple1  = (10, 20, 30)
print(tuple1)
print(type(tuple1))

tuple1[0] = 50
print(tuple1[0:2])
print(tuple1[2])

#access value of list
for x in tuple1:
    print(x+10)

#access index and value together
for i,x in enumerate(tuple1):
    print(i, x)
    
#loops are inefficient instead use functional operators like apply, map, etc.,

tuple2 = (10, True, 'abc')

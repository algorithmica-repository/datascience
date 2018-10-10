#tuple is an (immutable) indexed group of elements
tuple1 = (10, 20, 30, True, 23.6, 'abc')
print(type(tuple1))
print(tuple1)

tuple2 = ((10,20), 'def',(10,True))
print(len(tuple2))
print(tuple2[0])

#element access
print(tuple1[2])
print(tuple1[1:3])
print(tuple1[:3])
print(tuple1[2:])
print(tuple1[0:4:2])

#tuple traversal with for loop
for x in tuple1:
    print(x)
for i,x in enumerate(tuple1):
    print(i,x)
    
#tuple creation with comprehension(not supported)
tuple3 = (10,20,30) 
tuple4 =(x+10 for x in tuple3)

# no need to think of shallow vs deep copying


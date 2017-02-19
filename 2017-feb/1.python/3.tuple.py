#tuple: immutable list

tuple1 = (10, 20, 30, 40, 50)
type(tuple1)
tuple1[0]
tuple1[-1]
tuple1[0:3]
tuple1[2] = 100

for x in tuple1:
    print x ** 2
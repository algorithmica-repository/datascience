#creating lists
list1 = [10, 20, 30, 40, 50]
print(list1)
print(type(list1))

list2 = list([10,20,30.6])
print(list2)
print(type(list2))

list3 = [10.5, 20, True,"abc"]
print(list3)
print(type(list3))
list3.sort()

#access list elements by slicing
print(list1[0])
print(list1[0:3])
print(list1[0:])
print(list1[2:3])
print(list1[-1])

print(len(list1))

#method invocation
list1.append(60)
list1.append(5)
print(list1)
list1.pop()
list1.reverse()

list4 = [[1,2,3], 10, 50]
print(list4)
list4[0]

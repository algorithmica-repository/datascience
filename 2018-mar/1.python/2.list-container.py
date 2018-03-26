
#list is an indexed container that holds heterogeneous elements
list1 = [10, 30, 20, 40]
print(type(list1))
print(list1)

list2 = list(range(1, 10, 2))
print(list2)
print(len(list2))

#sliced access to elements of list
print(list1[0])
print(list1[0:2])
print(list1[0:4:2])
print(list1[0:])
print(list1[:3])
print(list1[0::2])
print(list1[-1])

list3 = [10, True, 'abc', [40,50]]
print(list3)

#modifying the contents of list
list4 = []
print(list4)
list4.append(10)
list4.append(20)
list4.insert(0,70)
list4.append(True)
print(list4.pop())

#sort the elements of list1
list1.sort()
print(list1)
list3.sort()
print(list3)

#iterate through the list elements
for x in list1:
    print(x+20)
    print(x)
for i,x in enumerate(list1):
    print(i, x)


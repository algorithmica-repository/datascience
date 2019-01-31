list1 = [10, 20, 30, 40]
print(list1)
print(type(list1))

list2 = [10, 30.6, True, 'abc', [10,30]]
print(list2)
print(type(list2))

#get the list elements
print(list2[4])
print(list2[0:3])
print(list2[1:])
print(list2[:3])

#list manipulations
list1[0] = 60
list1.append(100)
list1.insert(2, 80)
list1.reverse()
list1.pop()

#loops on list
for x in list1:
    print(x)
for i,x in enumerate(list1):
    print(i,x)



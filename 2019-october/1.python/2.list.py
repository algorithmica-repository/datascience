list1 = [10, 20, 'abc', True, 30.6]
print(type(list1))
print(list1)
print(id(list1))

list2 = ['xyz', 'a', "abc", 12]
print(list2)
print(id(list2))

list3 = list2
print(list3)
print(id(list3))

list4 = list2.copy()
print(list4)
print(id(list4))

print(list1 == list2)
print(list2 == list3)

list5 = [10, list1, 'abc', True, list4]
print(list5)
print(type(list5))

#element access
list1[2]
list1[1:4]
list1[1:4:2]
list1[2:]
list1[:3]

#manipulate list
list1[0] = 30
list1.append(50)
list1.pop()

#enumeration over list
for x in list1:
    print(x)

for i,x in enumerate(list1):
    print(i, x)

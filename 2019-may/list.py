#list: An indexed container of data
list1 = [10, 20, 34.6, True, 'abc']
print(list1)
print(type(list1))

print(list1[0])
print(list1[-1])

#subsetting of list elements
print(list1[0:4])
print(list1[1:])
print(list1[:4])
print(list1[-3:])

list1.append(False)
list1.reverse()
list1.pop()

list2 = [30, list1]
print(list2)
print(list2[1][0])

for x in list1:
    print(type(x))
    print(x)
    
for i,x in enumerate(list1):
    print(i, x)

    

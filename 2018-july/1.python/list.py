#list is an indexed container
#data items could be heterogeneous
##creation of list
list1 = [10, 30, 20, 40]
print(type(list1))
print(list1)

list2 = list(range(1,10,2))
print(type(list2))
print(list2)

## accessing list elements
list1[0]
list1[0:3]
list1[0:]
list1[:3]
list1[0::2]

#manipulate list 
list1[0] = 100
list3 = []
print(list3)
list3.append(10)
list3.append(20)
list3.insert(0,70)
list3.append(True)
list3.append(list1)

list1.sort()
list3.sort()
print(list1.pop())

print(len(list3))

#access value of list
for x in list2:
    print(x+10)

#access index and value together
for i,x in enumerate(list2):
    print(i, x)
    
#loops are inefficient instead use functional operators like apply, map, etc.,


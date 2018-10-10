#list ia an indexed group of elements
#elements could be heterogenous
list1 = [10,10.5,'abc',True]
print(type(list1))

list2 = [[10,20], 'def',[10,True]]
print(len(list2))
print(list2[0])

#element access
print(list1[2])
print(list1[1:3])
print(list1[:3])
print(list1[2:])
print(list1[0:4:2])

#list mutations
list1.append(100)
list1.insert(0,50)
list1.reverse()
list1.pop()

#list traversal with for loop
for x in list1:
    print(x)
for i,x in enumerate(list1):
    print(i,x)
    
#list creation with comprehension
list3 = [10,20,30] 
list4 = [x+10 for x in list3]

#shallow vs deep copying
list5 = list4
list6 = list4.copy()

list5[0] = 1000
list6[0] = 200

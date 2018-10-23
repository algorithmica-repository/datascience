#set is a group of distinct values
set1 = {10, 20, 30}
print(type(set1))
print(set1)

#individual set operations
print(len(set1))
print(10 in set1)

#multiple set level operations
set2 = { 30, 40, 50}
set3 = set1.union(set2)
print(set3)

set4 = set1.intersection(set2)
print(set4)

set5 = set1.difference(set2)
print(set5)

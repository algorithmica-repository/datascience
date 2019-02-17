import math

def entropy(n1, n2):
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    if(p1 == 0 or p2 == 0):
        return 0
    return -p1*math.log2(p1) - p2*math.log2(p2)

print(entropy(10, 0))
print(entropy(9, 1))
print(entropy(7, 3))
print(entropy(6, 4))
print(entropy(5, 5))

def gini_index(n1, n2):
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    return 1 - (p1*p1 + p2*p2)

print(gini_index(10, 0))
print(gini_index(9, 1))
print(gini_index(7, 3))
print(gini_index(6, 4))
print(gini_index(5, 5))

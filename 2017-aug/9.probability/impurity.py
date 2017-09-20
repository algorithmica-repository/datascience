import math
def entropy(p1,p2):
    return - p1 * math.log2(p1) - p2 * math.log2(p2)

print(entropy(10/20, 10/20))
print(entropy(12/20, 8/20))
print(entropy(15/20, 5/20))
print(entropy(19/20, 1/20))
print(entropy(20/20, 0/20))

def gini(p1,p2):
    return p1 * p2 + p2 * p1

print(gini(10/20, 10/20))
print(gini(12/20, 8/20))
print(gini(15/20, 5/20))
print(gini(19/20, 1/20))
print(gini(20/20, 0/20))

def gini(p1,p2):
    return 1 - p1 * p1 - p2 *p2
print(gini(10/20, 10/20))
print(gini(12/20, 8/20))
print(gini(15/20, 5/20))
print(gini(19/20, 1/20))
print(gini(20/20, 0/20))

print(gini(15/24,9/24))

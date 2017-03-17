import math
def entropy(p1,p2):
    return - p1 * math.log2(p1) - p2 * math.log2(p2)

entropy(19/20, 1/20)
entropy(15/20, 5/20)
entropy(10/20, 10/20)
entropy(12/20, 8/20)
entropy(20/20, 0/20)

def gini(p1,p2):
    return p1 * p2 + p2 * p1

gini(19/20, 1/20)
gini(15/20, 5/20)
gini(10/20, 10/20)
gini(12/20, 8/20)
gini(16/20, 4/20)
gini(20/20, 0/20)

def gini2(p1,p2):
    return 1 - p1 * p1 - p2 *p2
gini2(19/20, 1/20)
gini2(549/891,342/891)
gini(549/891,342/891)
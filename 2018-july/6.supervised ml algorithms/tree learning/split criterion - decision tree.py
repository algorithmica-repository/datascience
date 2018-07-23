import math

def entropy(n1, n2):
    if n1 == 0 or n2 == 0:
        return 0
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    return -p1*math.log2(p1) - p2*math.log2(p2)

print(entropy(10, 0))
print(entropy(9, 1))
print(entropy(7, 3))
print(entropy(6, 4))
print(entropy(5, 5))

def gini_index(n1, n2):
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    return 1-(p1*p1 + p2*p2)

print(gini_index(10, 0))
print(gini_index(9, 1))
print(gini_index(7, 3))
print(gini_index(6, 4))
print(gini_index(5, 5))

import matplotlib.pyplot as plt
import numpy as np

def gini(p):
   return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

def entropy(p):
   return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def classification_error(p):
   return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]
c_err = [classification_error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)

for j, lab, ls, c in zip(
      [ent, gini(x), c_err],
      ['Entropy', 'Gini Impurity', 'Misclassification Error'],
      ['-', '--', '-.'],
      ['lightgray', 'green', 'blue']):
   line = ax.plot(x, j, label=lab, linestyle=ls, lw=1, color=c)

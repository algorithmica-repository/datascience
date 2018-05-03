l1 = [10, 20, 30, 40]
l2 = []
for x in l1:
    l2.append(x+10)
l3 = [x+10  for x in l1]

d1 = {'abc':10, 'def':20, 'xyz':30}
d2 = {} 
for k in d1.keys():
    d2[k] = d1.get(k) + 10
d3 = {k:d1[k]+10  for k in d1.keys()}
    


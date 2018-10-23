map1 =  { 'depth':10, 'leaf_nodes':10, 'criteria':'gini'}
print(type(map1))
print(map1)

map2 =  { 'depth':(10,40), 'leaf_nodes':[10,20], 'criteria':['gini','entropy']}
print(type(map2))
print(map2)

map3 =  { 'depth':{ 'x':10,
                     'y':40
                    }, 
          'leaf_nodes':[10,20], 
          'criteria':['gini','entropy']}
print(type(map3))
print(map3)

#access elements by key
print(map1['depth'])
print(map1['depth1'])
print(map1.get('depth'))
print(map1.get('depth1'))

#delete the pair with given key
map1.pop('depth')

#add new key-value pair
map1['depth1'] = 50
map1['leaf_nodes'] = 60

#enumerate keys/values/items
print(map1.keys())
print(map1.values())
print(map1.items())

for x in map1.keys():
    print(x, map1.get(x))

for x in map1.items():
    print(x)


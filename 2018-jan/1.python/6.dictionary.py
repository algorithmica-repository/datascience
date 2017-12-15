parameters =  { 'depth':10, 'leaf_nodes':10}
print(type(parameters))

#access elements by key
print(parameters['depth'])
print(parameters['depth1'])
print(parameters.get('depth'))
print(parameters.get('depth1'))

a = None
print(type(a))

#add new key-value pair
parameters['depth1'] = 50
parameters['a'] = 10
parameters['a'] = 20

#enumerate over keys/values/items
print(parameters.keys())
print(parameters.values())
print(parameters.items())

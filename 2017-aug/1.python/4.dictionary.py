parameters =  { 'depth':10, 'leaf_nodes':10}
type(parameters)

#access elements by key
print(parameters['depth'])
print(parameters['depth1'])
print(parameters.get('depth'))
print(parameters.get('depth1'))

a = None
type(a)

parameters['depth1'] = 50
print(parameters.keys())
print(parameters.items())

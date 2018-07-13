##dictionary is a group of key-value pairs

#creating dictionaries
dictionary1 = {'abc':10, 'def':20, 'xyz':50}
print(type(dictionary1))
print(dictionary1)

dictionary2 = {'abc':[10,20,30], 'def':(20,30), 'xyz':50}
print(dictionary2)

#retrieve value of a specific key
print(dictionary2['abc'])
print(dictionary2['def'])
print(dictionary2.get('abc'))

print(dictionary2['de'])
print(dictionary2.get('de'))

#retrieve all keys, values & items
print(dictionary2.keys())
print(dictionary2.values())
print(dictionary2.items())


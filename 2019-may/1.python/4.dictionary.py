#dictionary: a set of key-value pairs(keys must be distinct)

persons = {'name':['abc','def','xyz'],
           'age':[30, 25, 15],
           'fare':[12.5, 40.3, 50]
           }
print(persons)
print(type(persons))

#access of elements
print(persons['name'])
print(persons.get('name'))
persons['id'] = [1,5,7]
persons['name'][2] = 'aaaa'
persons['name'].append('axy')

print(persons.keys())
print(persons.values())
print(persons.items())

print(id(persons))
persons1 = persons
print(id(persons1))

persons2 = persons.copy()
print(id(persons2))

persons.pop('fare')

#access dictionary
for x in persons2.keys():
    print(x, persons2.get(x))

for x in persons2.items():
    print(x)
    print(x[0], x[1])
      

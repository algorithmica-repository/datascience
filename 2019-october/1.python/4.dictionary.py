#dictionary: a set of key-value pairs(keys must be distinct)

persons = {'name':['abc','def','xyz'],
           'age':[30, 25, 15],
           'fare':[12.5, 40.3, 50]
           }
print(persons)
print(type(persons))
print(id(persons))

persons2 = persons.copy()
print(id(persons2))

#access of elements
print(persons['name'])
print(persons.get('name'))
print(persons['survived'])
print(persons.get('survived'))

#manipulation of content
persons['id'] = [1,5,7]
persons['dummy'] = (3,6)
persons['name'].append('axy')
persons.pop('fare')

#slicing dictionary
print(persons.keys())
print(persons.values())
print(persons.items())

#access dictionary
for x in persons2.keys():
    print(x, persons2.get(x))

for x in persons2.items():
    print(x)
    print(x[0], x[1])
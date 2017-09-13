name = "abcde"
print(type(name))

#access string content
print(name[0])
print(name[2:5])

#modify string content
name[0] = 'A'
name + 'xyz'
name = name + 'xyz'
print(name)
name = name.upper()
print(name)

name = name.replace('AB','pq')
print(name)

isinstance(name, str)
isinstance(name, int)








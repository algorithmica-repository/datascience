map1 = { "key1":10, "key2":20, "key3":30 }
print(map1)

map2 = {'10':['movie1','movie2'],
        '20':['movie3']}
print(map2)

#two ways of accessing data from map
map1.get("key3")
map1["key1"]
#get doesnt throw error when key is not present
map1.get('key4')
map1['key4']

#replace value of a key
map1["key1"] = 70
#add new key-value pair
map1["key5"] = 90

#key based access
for x in map1.keys():
    print(x, map1.get(x))
#item based access
for x in map1.items():
    print(x)
print(map1.values())

#duplicate keys never allowed in dictionary
map3 = { "key1":10, "key2":20, "key1":30 }


#dictionary/map/associative array: a group of key-value pairs 
passengers = {100:"abc",200:"def",300:"xyz"}
type(passengers)

map1 = { "key1":10, "key2":20, "key3":30 }
print(map1)

#two ways of accessing data from map
print(map1.get("key3"))
print(map1["key3"])

#get return none if key doesnot exist
print(map1.get("key4"))
#index operator throws error if key doesnot exist
print(map1["key4"])

#adding new key-value pair
map1["key4"] = 70

#replacing value for existing key
map1["key2"] = 90

print(map1.keys())
#iterate through keys of dictionary
for x in map1.keys():
    print(x, map1.get(x))
    
print(map1.items())

for x,y in map1.items():
    print(x, y)

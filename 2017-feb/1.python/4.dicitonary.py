#dictionary/map/associative array: a group of key-value pairs 
passengers = {100:"abc",200:"def",300:"xyz"}
type(passengers)

map1 = { "key1":10, "key2":20, "key3":30 }
        
print map1

#two ways of accessing data from map
map1.get("key4")
map1["key4"]

#adding new key-value pair
map1["key4"] = 70

#replacing value for existing key
map1["key2"] = 90

# If key doesnot exist in map, get does not throw exception whereas indexed access throws exception
map1.get("key7")
map1["key7"]

#iterate through keys of dictionary
for x in map1.keys():
    print x, map1.get(x)
    
type(map1.iteritems())

#iterate through the key-value items of dictionary
for x in map1.iteritems():
    print x

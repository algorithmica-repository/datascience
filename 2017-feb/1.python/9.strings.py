#strings are immutable
s1 = "abcdedf"
print(type(s1))
s1[0:3]
s1[0:]
s1[:5]
s1[::2]

s2 = s1.replace("ab","xy")
print(s2)
s3 = s1.capitalize()
print(s3)

print(isinstance(s3, str))
print(isinstance(s3, int))

#convert string to list of characters
s4 = list(s1)
s4[0] = 'x'

#list is converted to string(incorrect)
s5 = str(s4)
type(s5)

#correct way of converting list to string
s6 = ''.join(map(str,s4))
type(s6)

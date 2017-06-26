s1 = "abcdedf"
print(type(s1))

#strings are immutable
s1[0] = 'F'

#slice the contents of string
s1[0:3]
s1[0:]
s1[:5]
s1[::2]

s2 = s1.replace("ab","xy")
print(s1)
print(s2)
s3 = s1.capitalize()
print(s3)

print(isinstance(s3, str))
print(isinstance(s3, int))
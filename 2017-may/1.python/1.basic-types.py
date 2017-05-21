#python is dynamically typed language. the type of the variable is 
#inferred based on the value assigned to a variable at that time

#int type
x = 15
print type(x)

#float type
y = 15.6
print type(y)

#bool type
z = True
print type(z)

#complex type
c = 2  + 1j
print type(c)

#check the object type against target type
isinstance(x,int) 

#cast float to int
c1 = int(y)

#cast complext to int which is invalid
c2 = int(c)

#None is special value in python
p3 = None
type(p3)



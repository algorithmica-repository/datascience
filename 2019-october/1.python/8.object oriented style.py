#logic is arrange as group of object interactions

#goals of oo style: simplified maintenance, extensibility, reuse

#oo concepts supported by languages:
#class, object, inheritance, composition,polymorphism, dynamic binding, encapsulation, abstraction

#class allows us to create user defined types

#creating classes
class Dummy1():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def add(self):
        return self.a + self.b
    def mul(self):
        return self.a * self.b
    
print(help(Dummy1))    
d1 = Dummy1(10, 20)
print(d1.add())
print(d1.mul())

d2 = Dummy1(30, 40)
print(d2.add())
print(d2.mul())

print(d1.__dict__)
print(d2.__dict__)

#inheritance
class A():
    def __init__(self, a):
        self.a = a
    def display(self):
        return self.a

class B(A):
    def __init__(self, a, b):
        super().__init__(a)
        self.b = b
    def display(self):
        return self.a, self.b
        
class C(B):
    def display(self):
        return self.a+10, self.b+10

a = A(10)
print(a.__dict__)
print(a.display())

b = B(10, 20)
print(b.__dict__)
print(b.display())

c = C(30, 40)
print(c.__dict__)
print(c.display())

#check the type of object at runtime
print(type(a) == A)
print(isinstance(a, A))    

print(type(b) == A)
print(isinstance(b, A))   
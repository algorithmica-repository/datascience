#logic is arrange as group of object interactions

#goals of oo style: simplified maintenance, extensibility, reuse

#oo concepts supported by languages:
#class, object, inheritance, composition,polymorphism, dynamic binding, encapsulation, abstraction

#class allows us to create user defined types

class Dummy():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def add(self):
        return self.a + self.b
    def mul(self):
        return self.a * self.b

d1 = Dummy(10, 20)
print(id(d1))
print(d1.add())
print(d1.mul())
print(d1.a)

d2 = Dummy(2, 3)
print(id(d2))
print(d2.add())
print(d2.mul())
print(d2.a)

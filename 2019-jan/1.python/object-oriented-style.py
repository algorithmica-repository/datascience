#logic is shared across objects
#objects are designed to have data+methods

#encapsulation, polymorphism, inheritance are poweful tools to
#get reuse, extensibilty and simplified maintenance of software

class Pair():
    def __init__(self, a, b):
        self.a = a
        self.b = b
        c = 10
    def add(self):
        return self.a + self.b
    def subtract(self):
        return self.a - self.b
    def dummy(self):
        a = 20
        return a + 10
    
    
p1 = Pair(10, 20)
print(p1.add())
p2 = Pair(30, 40)
print(p2.add())    

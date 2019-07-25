def add(a, b) :
   return a+b

def mul(a, b):
    return a*b

class Complex():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def display(self):
        print(self.a , self.b)
        
#print(__name__)
if __name__ == '__main__':
    print(add(10,20) )
    print(mul(5, 10))
    c = Complex(10, 15)
    c.display()

class Dummy():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def add(self):
        return self.a + self.b
d1 = Dummy(10, 20)
print(d1.add())
d2 = Dummy(30, 40)
print(d2.add())

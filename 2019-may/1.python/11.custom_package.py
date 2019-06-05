def add(* args):
    res= 0
    for x in args:
        res = res + x
    return res

class Test():
    def __init__(self, a):
        self.a = a
    def display(self):
        print(self.a)
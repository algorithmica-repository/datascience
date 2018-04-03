def add(a, b):
    return a+b

print(add(10,20))

def add_mul(a,b,c=10,d=20):
    tmp1 = a + b
    tmp2 = c + d
    return tmp1 * tmp2
add_mul(1,2,3,4)
add_mul(1,2,3)
add_mul(1,2,d=3)

import sys
import custom_package as cp

print(sys.path)
print(cp.add(1,2,3))
t = cp.Test(10)
t.display()
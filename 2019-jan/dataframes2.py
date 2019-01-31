import pandas as pd

tmp = {'c1':[1,2,3,4] , 
       'c2':['abc','def','xyz','pqr'], 
       'c3':[10.3, 40.5, 60.7,70]
       }

df = pd.DataFrame(tmp)
print(df)

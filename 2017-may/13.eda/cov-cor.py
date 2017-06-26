import numpy as np
import pandas as pd
import seaborn as sns

v1 = [10,12,15,20,22,25]
v2 = [20,22,25,27,32,35]
v3 = [35,32,30,29,27,26]
v4 = [30,25,35,42,60,33]
np.cov(v1,v2)
np.cov(v1,v3)
np.cov(v1,v4)

v5 = [1000,1200,1500,2000,2200,2500]
v6 = [2000,2200,2500,2700,3200,3500]
np.cov(v5,v6)

df = pd.DataFrame({'v1':v1,'v2':v2,
'v3':v3,'v4':v4,'v5':v5, 'v6':v6})

sns.jointplot(x="v1", y="v2", data=df)
sns.jointplot(x="v1", y="v3", data=df)
sns.jointplot(x="v4", y="v5", data=df)
sns.jointplot(x="v1", y="v4", data=df)

#magnitude of cov reflects??
#sign reflects the type of relationship
print(np.corrcoef(v1,v2))
print(np.corrcoef(v1,v3))
print(np.corrcoef(v4,v5))

#z-scores
for e in v1:
    print((e - np.mean(v1))/np.std(v1))
for e in v6:
    print((e - np.mean(v6))/np.std(v6))

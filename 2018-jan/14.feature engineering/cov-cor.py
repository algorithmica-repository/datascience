import seaborn as sns
import pandas as pd
import numpy as np

ms_stocks = [10,40,30,40]
google_stocks = [20,30,40,50]
ibm_stocks = [60,50,40,30]
df = pd.DataFrame({'ms_stocks':ms_stocks, 'google_stocks':google_stocks,'ibm_stocks':ibm_stocks})
sns.jointplot(x='ms_stocks', y='google_stocks', data=df)
sns.jointplot(x='google_stocks', y='ibm_stocks', data=df)

input = np.array(df)
print(np.cov(input.T))
print(np.corrcoef(input.T))

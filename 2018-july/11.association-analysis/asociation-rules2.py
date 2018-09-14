import pandas as pd
from mlxtend import preprocessing 
from mlxtend import frequent_patterns 
import os
import csv

file = open(os.path.join('C:\\Users\\Algorithmica\\Downloads\\dataset\\75000','75000-out1.csv'))
rows = csv.reader(file)
rows = list(rows)
for row in rows:
    row.pop(0)

oht = preprocessing.OnehotTransactions()
oht.fit(rows)
print(oht.columns_)
oht_ary = oht.transform(rows)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
df

frequent_itemsets = frequent_patterns.apriori(df, min_support=0.05, use_colnames=True)

rules = frequent_patterns.association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
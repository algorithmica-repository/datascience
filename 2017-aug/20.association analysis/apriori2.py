import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori, association_rules
import os
import csv

os.chdir('D:/')

file = open('5000-out1.csv') 
rows = csv.reader(file)
rows = list(rows)
for row in rows:
    row.pop(0)

oht = OnehotTransactions()
oht_ary = oht.fit(rows).transform(rows)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
df

apriori(df, min_support=0.05)
apriori(df, min_support=0.05, use_colnames=True)

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]

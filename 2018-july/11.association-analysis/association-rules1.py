import pandas as pd
from mlxtend import preprocessing 
from mlxtend import frequent_patterns 

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

oht = preprocessing.OnehotTransactions()
oht.fit(dataset)
print(oht.columns_)
oht_ary = oht.transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
df

frequent_patterns.apriori(df, min_support=0.6)
frequent_itemsets = frequent_patterns.apriori(df, min_support=0.6, use_colnames=True)

frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.8) ]
                   

rules = frequent_patterns.association_rules(frequent_itemsets, metric="confidence", min_threshold=1)
rules[ (rules['lift'] > 1) &
       (rules['confidence'] >= 0.8) ]
import pandas as pd
import seaborn as sns
print(sns.__version__)

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

#continuous vs continuous relationship
sns.jointplot(x="Age", y="Fare", data=titanic_train)

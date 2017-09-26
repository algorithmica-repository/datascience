import pandas as pd
import seaborn as sns
import os

#changes working directory
os.chdir("D:/titanic")

titanic_train = pd.read_csv("train.csv")

titanic_train.shape
titanic_train.info()

#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)

#creaate categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age > 10 and age <= 25): 
        return 'Young'
    elif(age > 25 and age <= 50): 
        return 'Middle'
    else: 
        return 'Old'

titanic_train['Age1'] = titanic_train['Age'].map(convert_age)

sns.factorplot(x="Age1", hue="Survived", data=titanic_train, kind="count", size=6)

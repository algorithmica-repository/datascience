import pandas as pd
import seaborn as sns
import os
from sklearn import preprocessing

os.chdir('C:/Users/Algorithmica/Downloads')

#read and explore data
titanic_train = pd.read_csv('titanic_train.csv')
titanic_train.shape
titanic_train.info()

#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)
sns.factorplot(x="Title", hue="Survived", data=titanic_train, kind="count", size=6)

age_imputer = preprocessing.Imputer()
age_imputer.fit(titanic_train[['Age']])
titanic_train[['Age']] = age_imputer.transform(titanic_train[['Age']])

#creaate categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 10): 
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
titanic_train['Age1'] = titanic_train['Age'].map(convert_age)
sns.factorplot(x="Age1", hue="Survived", data=titanic_train, kind="count", size=6)

titanic_train['FamilySize'] = titanic_train['SibSp'] +  titanic_train['Parch'] + 1
titanic_train['FamilySize'].describe()
sns.FacetGrid(titanic_train, row="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic_train['FamilySize1'] = titanic_train['FamilySize'].map(convert_familysize)
sns.factorplot(x="FamilySize1", hue="Survived", data=titanic_train, kind="count", size=6)
           


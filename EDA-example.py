# disable warnings
import warnings

warnings.simplefilter("ignore")

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()    # visualization lib
import os  
print(os.listdir('../input/ch01-titanic'))

train_df = pd.read_csv('../input/ch01-titanic/train.csv')
test_df = pd.read_csv('../input/ch01-titanic/test.csv')
sub = pd.read_csv('../input/ch01-titanic/gender_submission.csv')

pd.set_option('display.max_columns', None)  # show all the colunms
print(train_df.head())  # show first 5 datasets

print("train data size: ", train_df.shape)
print("test data size: ", test_df.shape)

print("show if any null data in train data: ",train_df.isnull().sum())
print("show if any null data in test data: ",test_df.isnull().sum())

# there are missing data in 'Age' and 'Cabin' and a little in 'Embarked' column
# imputation : handling the missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean") # use mean instead for missing data

train_df['Age'] = imputer.fit_transform(np.array(train_df['Age']).reshape(891,1))
train_df.Embarked.fillna(method='ffill', inplace=True)  # use the right 1 previous data instead for missing data (forward fill)
train_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True) # we don't need these datasets. axis=1: not row but column, inplace=True: do not create new frames instead

# do the same to testdata
test_df['Age'] = imputer.fit_transform(np.array(test_df['Age']).reshape(418, 1))
test_df.Embarked.fillna(method='ffill', inplace=True)
test_df.Fare.fillna(method='ffill', inplace=True)
test_df.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)

# data visualization
sns.countplot(x='Survived', hue='Sex', data=train_df)
#plt.show()

sns.countplot(x='Embarked', hue='Survived', data=train_df)
#plt.show()

sns.countplot(x='SibSp', hue='Survived', data=train_df)
#plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train_df)
#plt.show()

plt.figure(figsize=(10, 5))
sns.distplot(train_df['Age'], bins=24, color='b')
plt.show()

numeric_df = train_df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
plt.title('Titanic Correlation of Features', y=1.05, size=15)
sns.heatmap(numeric_df.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:04:23 2021

@company: DEVINN s.r.o
@author: trdla
"""
import pandas as pd
import numpy as np

# Load data
gender_submission = pd.read_csv("gender_submission.csv") 
train = pd.read_csv("train.csv") 
test = pd.read_csv("test.csv") 

# Exploratory Analysis
train.head()

# explore nans
train.isna().sum()
test.isna().sum()

# merge datasets for some exploration
df_merged = train.append(test).reset_index(drop=True)

# Feature Engineering
# remove name and Ticket
train.drop(['Name', 'Ticket'], axis=1, inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)

# solve missing values

# cabins contain block letter, check what is available
import string
for letter in string.ascii_lowercase:
    if not any(df_merged['Cabin'].dropna().str.contains(letter.capitalize())):
        print(letter)
# 'Z' is free
train['Cabin'].fillna('Z', inplace=True)
test['Cabin'].fillna('Z', inplace=True)

# embarked
df_merged['Embarked'].unique()
train['Embarked'].fillna('A', inplace=True)
test['Embarked'].fillna('A', inplace=True)

# Fare
train['Fare'].fillna(0, inplace=True)
test['Fare'].fillna(0, inplace=True)

# create feature from cabin
def contained_letter(value):
    for letter in string.ascii_lowercase:
        if letter.capitalize() in value:
            return letter.capitalize()
    return 'Z'
train['Cabin'] = train['Cabin'].apply(lambda x: contained_letter(x))
test['Cabin'] = test['Cabin'].apply(lambda x: contained_letter(x))


# prepare one-hot-encoded columns
# In test set we wont know the "survived" feature and "PassengerId" is no use for imputation
df_train_dummy = pd.get_dummies(train.drop(['Survived','PassengerId'], axis=1))
df_test_dummy = pd.get_dummies(test.drop(['PassengerId'], axis=1))

# handle missing columns
df_test_dummy = df_test_dummy.reindex(columns=df_train_dummy.columns, fill_value=0)
df_train_dummy = df_train_dummy.reindex(columns=df_test_dummy.columns, fill_value=0)

# age - imputate
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance") # distance might be better than "uniform"
imputer.fit(df_train_dummy)
# train
train_dummy = pd.DataFrame(imputer.transform(df_train_dummy), columns=df_train_dummy.columns)
# test
test_dummy = pd.DataFrame(imputer.transform(df_test_dummy), columns=df_test_dummy.columns)

# Preprocessing
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# fit train
train_dummy.loc[:,['Age','Fare']] = normalizer.fit_transform(train_dummy.loc[:,['Age','Fare']])
# transform test
test_dummy.loc[:,['Age','Fare']] = normalizer.transform(test_dummy.loc[:,['Age','Fare']])

# Predict survival
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=30, learning_rate=0.3, max_depth=5, random_state=0)
clf.fit(train_dummy, train['Survived'])
clf.score(test_dummy, gender_submission['Survived'])

# Finetune
# prepare data so it does not need to be loaded everytime
gender_submission_static = pd.read_csv("gender_submission.csv") 
train_static = pd.read_csv("train.csv") 
test_static = pd.read_csv("test.csv") 
train_static.drop(['Name', 'Ticket'], axis=1, inplace=True)
test_static.drop(['Name', 'Ticket'], axis=1, inplace=True)

# define function
def opt_fnc(hp_knn_neighbors, hp_gbc_estimators, hp_gbc_learning_rate, hp_gbc_max_depth, predict=False):
    # copy source data from static
    train = train_static.copy(deep=True)
    test = test_static.copy(deep=True)
    
    # Feature Engineering
        # 'Z' is free
    train['Cabin'].fillna('Z', inplace=True)
    test['Cabin'].fillna('Z', inplace=True)
    
    # embarked
    train['Embarked'].fillna('A', inplace=True)
    test['Embarked'].fillna('A', inplace=True)
    
    # Fare
    train['Fare'].fillna(0, inplace=True)
    test['Fare'].fillna(0, inplace=True)
    
    # create feature from cabin
    def contained_letter(value):
        for letter in string.ascii_lowercase:
            if letter.capitalize() in value:
                return letter.capitalize()
        return 'Z'
    train['Cabin'] = train['Cabin'].apply(lambda x: contained_letter(x))
    test['Cabin'] = test['Cabin'].apply(lambda x: contained_letter(x))
    
    # prepare one-hot-encoded columns
    # In test set we wont know the "survived" feature and "PassengerId" is no use for imputation
    df_train_dummy = pd.get_dummies(train.drop(['Survived','PassengerId'], axis=1))
    df_test_dummy = pd.get_dummies(test.drop(['PassengerId'], axis=1))
    
    # handle missing columns
    df_test_dummy = df_test_dummy.reindex(columns=df_train_dummy.columns, fill_value=0)
    df_train_dummy = df_train_dummy.reindex(columns=df_test_dummy.columns, fill_value=0)
    
    # age - imputate
    imputer = KNNImputer(n_neighbors=hp_knn_neighbors, weights="distance") # distance might be better than "uniform"
    imputer.fit(df_train_dummy)
    # train
    train_dummy = pd.DataFrame(imputer.transform(df_train_dummy), columns=df_train_dummy.columns)
    # test
    test_dummy = pd.DataFrame(imputer.transform(df_test_dummy), columns=df_test_dummy.columns)
    
    # Preprocessing
    normalizer = Normalizer()
    # fit train
    train_dummy.loc[:,['Age','Fare']] = normalizer.fit_transform(train_dummy.loc[:,['Age','Fare']])
    # transform test
    test_dummy.loc[:,['Age','Fare']] = normalizer.transform(test_dummy.loc[:,['Age','Fare']])
    
    # Predict survival
    clf = GradientBoostingClassifier(n_estimators=hp_gbc_estimators, learning_rate=hp_gbc_learning_rate, max_depth=hp_gbc_max_depth, random_state=0)
    clf.fit(train_dummy, train['Survived'])
    
    if predict:
        predictions = pd.DataFrame(clf.predict(test_dummy), columns=['Survived'])
        predictions.loc[:,'PassengerId'] = test.loc[:,'PassengerId']
        return predictions
    else:
        return clf.score(test_dummy, gender_submission['Survived'])

# init optimizer
from skopt.optimizer import Optimizer
import skopt.space as HP_dtypes

knn_neighbors = HP_dtypes.Integer(low=2, high=10)
gbc_estimators = HP_dtypes.Integer(low=2, high=100)
gbc_learning_rate = HP_dtypes.Real(low=1e-5, high=1)
gbc_max_depth = HP_dtypes.Integer(low=1, high=10)

opt = Optimizer(dimensions=[knn_neighbors,
                            gbc_estimators,
                            gbc_learning_rate,
                            gbc_max_depth],
                acq_func="EI")
# run loop
for i in range(0,50):
    # get hps
    hps = opt.ask()
    # run fnc
    score = opt_fnc(*hps)
    print("Iteration #{0} score: {1}".format(i, score))
    # update optimizer
    opt.tell(x=hps, y=(1-score)) # (1-score) -> EI acq_func => trying to minimize it
    if score == 1:
        break
# retrieve best
best_hps = opt.get_result().x
# see the results
df_predicted = opt_fnc(*hps, predict=True)
df_predicted

from sklearn.metrics import classification_report
c_report = classification_report(gender_submission['Survived'], df_predicted['Survived'])
print(c_report)

# save the result
df_predicted.to_csv("submission.csv", sep=",", index=False)
































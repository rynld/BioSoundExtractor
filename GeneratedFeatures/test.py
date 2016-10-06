from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np

df = pd.read_csv('./rbm_mfcc.csv')
Y = pd.factorize(df.Category)[0]
X = df.drop(['Category'],axis=1).values


# mean = np.mean(X,axis=0)
# std = np.std(X,axis=0)
#
# for i in range(len(std)):
#     if std[i] == 0:
#         std[i] = 1

#print(df.describe())
#exit()
#Y = pd.get_dummies(Y).values


rf = RandomForestClassifier(n_estimators=100,oob_score=True)
rf.fit(X,Y)
print(rf.oob_score_)


kfold = KFold(len(Y),5,shuffle=True)
error = []
for idx_train,idx_test in kfold:
    lr = LogisticRegression()
    lr.fit(X[idx_train,:],Y[idx_train])
    pred = lr.predict(X[idx_test,:])

    error.append(accuracy_score(Y[idx_test],pred))

print(np.mean(error))
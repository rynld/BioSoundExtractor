
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    """
    Arguments
    ---------
    n_estimator: integer, optional
        Number of trees in the forest. Default is set to 100

    max_depth: interger or None,optional
        Depth of each tree in the forest, if None the tree is expanded completed

    min_sample_split: integer,optional
        Minimum number of sample need to make a split

    max_features: int,float,string or None, optional
        The number of features to consider for make a split

        -int the max number of features is equal to de value
        -float the max number of features is equal a percentage of n_features
        -"auto" the max number of features is equal to sqrt(n_features)
        -"sqrt" the max number of features is equal to sqrt(n_features)
        -"log2" the max number of features is equal to log2(n_features)

    Attributes
    ----------

    oob_score: out of bag error

    variable_importance: importance of each variable

    """
    def __init__(self,n_estimators=100,max_depth=None,min_sample_split=2,max_features='auto'):
        self.rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                         min_samples_split=min_sample_split,max_features=max_features
                                         ,oob_score=True)



    def train(self,X,Y):
        self.rf.fit(X,Y)

    def classify(self, X):
        return np.array(self.rf.predict(X))


    @property
    def oob_score(self):
        """
        :return: Return the out of bag error
        """
        return self.rf.oob_score_

    @property
    def variable_importance(self):
        """
        :return: array of float
            Importance of each variables, if higher more important
        """
        return self.rf.feature_importances_



import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin
import numpy as np


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers,
    like the SimpleImputer() or OrdinalEncoder(), to allow
    the use of the transformer on a selected group of variables.
    """

    def __init__(self, variables=None, transformer=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        The fit method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        # try:
        X[self.variables] = self.transformer.transform(X[self.variables])
        # except:
        #     print(len(self.variables))
        #     print((self.transformer.transform(X[self.variables])).shape)

        return X


class ChangeType(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_vars=None):
        self.numerical_vars = numerical_vars

    def fit(self, X, y=None):
        """
        The fit method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # Change the type of numerical features to int or float
        X = X.copy()
        for col in self.numerical_vars:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Set negative values to null

        for x in self.numerical_vars:
            X.loc[X[x] < 0, x] = np.nan

        return X


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        """
        The fit method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # drop unnecessary / unused features from the data set

        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)
        return X


class RemoveOutliers(TransformerMixin):
    def __init__(self, numerical_vars=None):
        self.numerical_vars = numerical_vars

    def fit(self, X, y):
        """
        The fit method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """
        self.X = X
        self.y = y
        return self

    def transform(self, X, y=None):
        # print(111)
        X = X.copy()

        # drop the outliers from the data set
        for x in self.numerical_vars:
            Q1 = self.X[x].quantile(0.25)
            Q3 = self.X[x].quantile(0.75)
            IQR = Q3 - Q1
            index_lb = self.X.loc[self.X[x] > (Q3 + 1.5 * IQR)].index
            self.X.drop(index_lb, inplace=True)
            self.y.drop(index_lb, inplace=True)
            index_ub = self.X.loc[self.X[x] < (Q1 - 1.5 * IQR)].index
            self.X.drop(index_ub, inplace=True)
            self.y.drop(index_ub, inplace=True)

        # return pd.concat([X,y], axis = 1)
        return self.X

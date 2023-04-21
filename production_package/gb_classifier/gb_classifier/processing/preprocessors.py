import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training data set.
        """

        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X

class ChangeType(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_vars=None):
        self.numerical_vars = numerical_vars

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # Change the type of numerical features to int or float
        X = X.copy()
        for x in X:
            X[x] = pd.to_numeric(X[x], errors='coerce')

        # Set negative values to null
        for x in X:
            for i in range(len(X[x])):
                if X[x][i] <= 0:
                    X[x][i] = None

        return X


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # drop unnecessary / unused features from the data set
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis=1)

        return X


class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_vars=None):
        self.numerical_vars = numerical_vars

    def fit(self, X, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """

        return self

    def transform(self, X):
        # drop the outliers from the data set
        X = X.copy()

        for x in X:
            Q1 = x.quantile(0.25)
            Q3 = x.quantile(0.75)
            IQR = Q3 - Q1
            X = X.drop(X.loc[x > (Q3 + 1.5 * IQR)].index)
            X = X.drop(X.loc[x < (Q1 - 1.5 * IQR)].index)

        return X

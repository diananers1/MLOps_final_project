import logging

from gb_classifier.config.core import config
from gb_classifier.processing import preprocessors as pp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

_logger = logging.getLogger(__name__)


score_pipe = Pipeline(
    [
        (
            "change_type",
            pp.ChangeType(
                numerical_vars=config.model_config.numerical_vars,
            ),
        ),

        (
            "numerical_imputer",
            pp.SklearnTransformerWrapper(
                variables=config.model_config.numerical_vars,
                transformer=SimpleImputer(strategy="mean"),
            ),
        ),
        (
            "categorical_imputer",
            pp.SklearnTransformerWrapper(
                variables=config.model_config.categorical_vars,
                transformer=SimpleImputer(strategy="most_frequent"),
            ),
        ),

        (
            "categorical_encoder",
            pp.SklearnTransformerWrapper(
                variables=config.model_config.categorical_vars,
                transformer=OrdinalEncoder(),
            ),
        ),

        (
            "remove_outliers",
            pp.RemoveOutliers(
                numerical_vars=config.model_config.numerical_vars,
            ),
        ),

        (
            "drop_features",
            pp.DropUnnecessaryFeatures(
                variables_to_drop=config.model_config.drop_features,
            ),
        ),

        (
            "gb_model",
            GradientBoostingClassifier(
                loss=config.model_config.loss,
                random_state=config.model_config.random_state,
                n_estimators=config.model_config.n_estimators,
                learning_rate=config.model_config.learning_rate,
                max_depth=config.model_config.max_depth
            ),
        ),
    ]
)

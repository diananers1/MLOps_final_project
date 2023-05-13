from gb_classifier.gb_classifier.config.core import config
from gb_classifier.gb_classifier.processing import preprocessors as pp


def test_drop_unnecessary_features_transformer(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    for column in  config.model_config.drop_features:
        assert column in X_train.columns

    transformer = pp.DropUnnecessaryFeatures(
        variables_to_drop=config.model_config.drop_features,
    )

    # When
    X_transformed = transformer.transform(X_train)

    # Then
    for column in config.model_config.drop_features:
        assert column not in X_transformed.columns

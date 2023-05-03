from ..gb_classifier import pipeline
from ..gb_classifier.config.core import config
from ..gb_classifier.processing.validation import validate_inputs


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    assert config.model_config.drop_features in X_train.columns
    pipeline.price_pipe.fit(X_train, y_train)

    # When
    # We access the transformed inputs with slicing
    transformed_inputs = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    assert config.model_config.drop_features in X_train.columns
    assert config.model_config.drop_features not in transformed_inputs.columns


def test_pipeline_predict_is_smaller_than_sample_testdata_input(pipeline_inputs, sample_test_data):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.price_pipe.fit(X_train, y_train)

    # When
    validated_inputs, errors = validate_inputs(input_data=sample_test_data)
    predictions = pipeline.price_pipe.predict(
        validated_inputs[config.model_config.features]
    )

    # Then
    assert len(predictions) <= len(sample_test_data)
    assert errors is None

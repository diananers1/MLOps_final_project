from gb_classifier.config.core import config
from gb_classifier.predict import make_prediction


# the test below is designed to protect us against gradual degradation across many model changes and updates
def test_classification_accuracy_against_benchmark(raw_training_data):
    # Given
    X = raw_training_data.drop(config.model_config.target, axis=1)
    y_true = raw_training_data[config.model_config.target]

    benchmark_accuracy = 0.5  # acceptable accuracy
    benchmark_classes = {0,1,2}  # set of all unique class labels

    # When
    subject = make_prediction(input_data=X[0:1])

    # Then
    assert subject is not None
    prediction = subject.get("predictions")[0]
    assert isinstance(prediction, int)
    assert prediction in benchmark_classes
    accuracy = sum(y_true == prediction) / len(y_true)
    assert accuracy >= benchmark_accuracy




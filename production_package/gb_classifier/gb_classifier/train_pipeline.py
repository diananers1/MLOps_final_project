import logging

from gb_classifier.version import VERSION as _version
from gb_classifier import pipeline
from gb_classifier.config.core import config
from gb_classifier.processing.data_management import (
    load_dataset,
    save_pipeline,
)
from sklearn.model_selection import train_test_split

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    pipeline.score_pipe.fit(X_train, y_train)

    _logger.warning(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.score_pipe)


if __name__ == "__main__":
    run_training()

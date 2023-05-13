import json
import numpy as np
import pytest
from gb_classifier.gb_classifier.processing.data_management import load_dataset

SECONDARY_VARIABLES_TO_RENAME = {
    "Num_of_Delayed_Payment": "DelayedPaymCount",
    "Credit_Utilization_Ratio": "CrdUR",
    "Outstanding_Debt": "OutDebt",
}


@pytest.mark.integration
def test_health_endpoint(client):
    # When
    response = client.get("/")

    # Then
    assert response.status_code == 200
    assert json.loads(response.data) == {"status": "ok"}


@pytest.mark.integration
@pytest.mark.parametrize(
    "api_endpoint, expected_no_predictions",
    (
        (
            "/creditscore/predict",
            50000,
        ),
    ),
)
def test_prediction_endpoint(api_endpoint, expected_no_predictions, client):
    # Given
    # Load the test dataset which is included in the model package
    test_inputs_df = load_dataset(file_name="test.csv")  # dataframe
    if api_endpoint == "/creditscore/predict":
        # adjust column names to those expected by the secondary model
        test_inputs_df.rename(columns=SECONDARY_VARIABLES_TO_RENAME, inplace=True)

    # When
    response = client.post(
        api_endpoint, json=test_inputs_df.to_dict(orient="records")
    )

    # Then
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["errors"] is None
    assert len(data["predictions"]) == expected_no_predictions


@pytest.mark.parametrize(
    "field, field_value, index, expected_error",
    (
        (
            "Monthly_Balance",
            "3",  # expected str
            33,
            {"33": {"Monthly_Balance": ["Not a valid integer."]}},
        ),
        (
            "Monthly_Inhand_Salary",
            "yes",  # expected integer
            45,
            {"45": {"Monthly_Inhand_Salary": ["Not a valid number."]}},
        ),
        (
            "Annual_Income",
            np.nan,  # nan not allowed
            34804,
            {"34804": {"Annual_Income": ["Field may not be null."]}},
        ),
    ),
)
@pytest.mark.integration
def test_prediction_validation(field, field_value, index, expected_error, client):
    # Given
    # Load the test dataset which is included in the model package
    test_inputs_df = load_dataset(file_name="test.csv")

    test_inputs_df.loc[index, field] = field_value

    # When
    response = client.post(
        "/creditscore/predict", json=test_inputs_df.to_dict(orient="records")
    )

    # Then
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data == expected_error

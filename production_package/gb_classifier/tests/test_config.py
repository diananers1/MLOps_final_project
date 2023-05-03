from pathlib import Path

import pytest
from ..gb_classifier.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
package_name: gb_classifier
training_data_file: credit_score.csv
test_data_file: test.csv
drop_features:
  - ID
  - Customer_ID
  - Month
  - Name
  - SSN
  - Type_of_Loan
  - Credit_History_Age
pipeline_name: gb_classification
pipeline_save_file: gb_classification_output_v
target: Credit_Score
features:
  - ID
  - Customer_ID
  - Month
  - Name
  - SSN
  - Type_of_Loan
  - Changed_Credit_Limit
  - Payment_of_Min_Amount
  - Credit_Mix
  - Delay_from_due_date
  - Annual_Income
  - Monthly_Inhand_Salary
  - Age
  - Monthly_Balance
  - Num_of_Delayed_Payment
  - Outstanding_Debt
  - Payment_Behaviour
  - Credit_History_Age
  - Num_Bank_Accounts
  - Credit_Utilization_Ratio
  - Occupation
  - Num_Credit_Card
  - Num_of_Loan
  - Total_EMI_per_month
  - Amount_invested_monthly
  - Interest_Rate
  - Num_Credit_Inquiries
numerical_vars:
  - Age
  - Annual_Income
  - Monthly_Inhand_Salary
  - Num_Bank_Accounts
  - Num_Credit_Card
  - Interest_Rate
  - Num_of_Loan
  - Delay_from_due_date
  - Num_of_Delayed_Payment
  - Changed_Credit_Limit
  - Num_Credit_Inquiries
  - Outstanding_Debt
  - Credit_Utilization_Ratio
  - Total_EMI_per_month
  - Amount_invested_monthly
  - Monthly_Balance
categorical_vars:
  - Occupation
  - Credit_Mix
  - Payment_of_Min_Amount
  - Payment_Behaviour
numerical_na_not_allowed:
  - Age
  - Annual_Income
  - Num_Bank_Accounts
  - Num_of_Delayed_Payment
  - Outstanding_Debt
  - Credit_History_Age
  - Monthly_Balance
test_size: 0.25
random_state: 42
n_estimators: 100
learning_rate: 0.2
max_depth: 5
loss: log_loss
allowed_loss_functions:
  - log_loss
  - deviance
  - exponential
"""


INVALID_TEST_CONFIG_TEXT = """
package_name: gb_classifier
training_data_file: credit_score.csv
test_data_file: test.csv
drop_features:
  - ID
  - Customer_ID
  - Month
  - Name
  - SSN
  - Type_of_Loan
  - Credit_History_Age
pipeline_name: gb_classification
pipeline_save_file: gb_classification_output_v
target: Credit_Score
features:
  - ID
  - Customer_ID
  - Month
  - Name
  - SSN
  - Type_of_Loan
  - Changed_Credit_Limit
  - Payment_of_Min_Amount
  - Credit_Mix
  - Delay_from_due_date
  - Annual_Income
  - Monthly_Inhand_Salary
  - Age
  - Monthly_Balance
  - Num_of_Delayed_Payment
  - Outstanding_Debt
  - Payment_Behaviour
  - Credit_History_Age
  - Num_Bank_Accounts
  - Credit_Utilization_Ratio
  - Occupation
  - Num_Credit_Card
  - Num_of_Loan
  - Total_EMI_per_month
  - Amount_invested_monthly
  - Interest_Rate
  - Num_Credit_Inquiries
numerical_vars:
  - Age
  - Annual_Income
  - Monthly_Inhand_Salary
  - Num_Bank_Accounts
  - Num_Credit_Card
  - Interest_Rate
  - Num_of_Loan
  - Delay_from_due_date
  - Num_of_Delayed_Payment
  - Changed_Credit_Limit
  - Num_Credit_Inquiries
  - Outstanding_Debt
  - Credit_Utilization_Ratio
  - Total_EMI_per_month
  - Amount_invested_monthly
  - Monthly_Balance
categorical_vars:
  - Occupation
  - Credit_Mix
  - Payment_of_Min_Amount
  - Payment_Behaviour
numerical_na_not_allowed:
  - Age
  - Annual_Income
  - Outstanding_Debt
  - Credit_History_Age
  - Monthly_Balance
test_size: 0.25
random_state: 42
n_estimators: 100
learning_rate: 0.2
max_depth: 5
loss: log_loss
allowed_loss_functions:
  - exponential
"""


def test_fetch_config_structure(tmpdir):  # tmpdir is a pytest built-in fixture
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    sample_config = configs_dir / "sample_config.yml"
    sample_config.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=sample_config)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    sample_config = configs_dir / "sample_config.yml"

    # invalid config attempts to set a prohibited loss
    # function which we validate against an allowed set of
    # loss function parameters.
    sample_config.write_text(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=sample_config)

    # When
    with pytest.raises(ValidationError) as e_info:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "not in the allowed set" in str(e_info.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    configs_dir = Path(tmpdir)
    sample_config = configs_dir / "sample_config.yml"

    TEST_CONFIG_TEXT = """package_name: gb_classifier"""
    sample_config.write_text(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=sample_config)

    # When
    with pytest.raises(ValidationError) as e_info:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "numerical_na_not_allowed" in str(e_info.value)
    assert "pipeline_name" in str(e_info.value)

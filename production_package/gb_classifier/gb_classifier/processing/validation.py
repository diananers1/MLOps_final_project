from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from gb_classifier.config.core import config
from marshmallow import fields, Schema, ValidationError


class CustomerInputSchema(Schema):
    Age = fields.Integer()
    Amount_invested_monthly = fields.Float(allow_none=True)
    Annual_Income = fields.Float()
    Changed_Credit_Limit = fields.Float(allow_none=True)
    Credit_History_Age = fields.Integer()
    Credit_Mix = fields.Str(allow_none=True)
    Credit_Utilization_Ratio = fields.Float(allow_none=True)
    Customer_ID = fields.Integer(allow_none=True)
    Delay_from_due_date = fields.Integer(allow_none=True)
    ID = fields.Integer(allow_none=True)
    Interest_Rate = fields.Integer(allow_none=True)
    Monthly_Balance = fields.Float()
    Monthly_Inhand_Salary = fields.Float(allow_none=True)
    Name = fields.Str(allow_none=True)
    Num_Bank_Accounts = fields.Integer()
    Num_Credit_Card = fields.Integer(allow_none=True)
    Num_Credit_Inquiries = fields.Integer(allow_none=True)
    Num_of_Delayed_Payment = fields.Integer()
    Num_of_Loan = fields.Integer(allow_none=True)
    Occupation = fields.Str(allow_none=True)
    Outstanding_Debt = fields.Float()
    Payment_Behaviour = fields.Str(allow_none=True)
    Payment_of_Min_Amount = fields.Str(allow_none=True)
    SSN = fields.Integer(allow_none=True)
    Total_EMI_per_month = fields.Float(allow_none=True)
    Type_of_Loan = fields.Str(allow_none=True)


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    if input_data[config.model_config.numerical_na_not_allowed].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.model_config.numerical_na_not_allowed
        )

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Check model inputs for unprocessable values."""

    # input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)  # TODO: Delete if you are sure.
    validated_data = drop_na_inputs(input_data=input_data)

    # set many=True to allow passing in a list
    schema = CustomerInputSchema(many=True)
    errors = None

    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as err:
        errors = err.messages

    return validated_data, errors

from gb_classifier.processing.validation import validate_inputs


def test_validate_inputs(sample_test_data):
    # When
    validated_inputs, errors = validate_inputs(input_data=sample_test_data)

    # Then
    assert not errors

    assert len(sample_test_data) == 50000
    assert len(validated_inputs) <= 50000


def test_validate_inputs_identifies_errors(sample_test_data):
    # Given
    test_inputs = sample_test_data.copy()

    # introduce errors
    test_inputs.at[1, "Annual_Income"] = "900000"  # we expect a float

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors
    assert errors[1] == {"Annual_Income": ["Not a valid float."]}

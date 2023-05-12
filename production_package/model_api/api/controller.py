import json

from flask import request, jsonify, Response
from gb_classifier.gb_classifier.predict import make_prediction
from prometheus_client import Histogram, Gauge, Info
from api.config import APP_NAME

PREDICTION_TRACKER = Histogram(
    name='house_price_prediction_dollars',
    documentation='ML Model Prediction on House Price',
    labelnames=['app_name', 'model_name', 'model_version']
)

PREDICTION_GAUGE = Gauge(
    name='house_price_gauge_dollars',
    documentation='ML Model Prediction on House Price for min max calcs',
    labelnames=['app_name', 'model_name', 'model_version']
)

PREDICTION_GAUGE.labels(
                app_name=APP_NAME,
                model_name=ModelType.LASSO.name,
                model_version=live_version)

MODEL_VERSIONS = Info(
    'model_version_details',
    'Capture model version information',
)

MODEL_VERSIONS.info({
    'live_model': ModelType.LASSO.name,
    'live_version': live_version,
    'shadow_model': ModelType.GRADIENT_BOOSTING.name,
    'shadow_version': shadow_version})

def health():
    if request.method == "GET":
        return jsonify({"status": "ok"})


def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        # Step 2: Access the model prediction function (also validates data)
        result = make_prediction(input_data=json_data)

        # Step 3: Handle errors
        errors = result.get("errors")
        if errors:
            return Response(json.dumps(errors), status=400)

        # Step 4: Split out results
        predictions = result.get("predictions").tolist()
        version = result.get("version")

        # Step 5: Prepare prediction response
        return jsonify(
            {"predictions": predictions, "version": version, "errors": errors}
        )

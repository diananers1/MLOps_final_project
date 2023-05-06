from flask import request, Flask  # request object encapsulates the HTTP request sent by the client
from flask.wrappers import Response
from prometheus_client import Counter, Histogram, Summary
import time

from api.config import APP_NAME

REQUEST_COUNT = Counter(
    name='request_counter',
    documentation='App Request Count',
    labelnames=['app_name', 'method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    name='request_latency_seconds',
    documentation='App Request latency',
    labelnames=['app_name', 'endpoint']
)

REQUEST_TIME = Summary(
    name="request_processing_seconds",
    documentation="Time spent processing request"
)


def start_timer() -> None:
    """Get start time of a request."""
    request._prometheus_metrics_request_start_time = time.time()


def stop_timer(response: Response) -> Response:
    """Get stop time of a request."""
    request_latency = time.time() - request._prometheus_metrics_request_start_time
    REQUEST_LATENCY.labels(
        app_name=APP_NAME,
        endpoint=request.path).observe(request_latency)
    return response


def record_request_data(response: Response) -> Response:
    """Capture request data.
    Uses the flask request object to extract information such as
    the HTTP request method, endpoint and HTTP status.
    """
    REQUEST_COUNT.labels(
        app_name=APP_NAME,
        method=request.method,
        endpoint=request.path,
        http_status=response.status_code).inc()
    return response


def setup_metrics(app: Flask) -> None:
    """Setup Prometheus metrics.
    This function uses the flask before_request
    and after_request hooks to capture metrics
    with each HTTP request to the application.
    """
    app.before_request(start_timer)
    app.after_request(record_request_data)
    app.after_request(stop_timer)

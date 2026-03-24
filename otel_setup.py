"""OpenTelemetry SDK initialization for faster-qwen3-tts."""

import contextvars
import logging
import os
from typing import Callable

from opentelemetry import trace, metrics

# Contextvar for task_id — set on WebSocket request, read by TaskIdFilter
# to inject into all log records for GCP Cloud Logging search (labels.task_id="xxx").
current_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_task_id", default="")


class TaskIdFilter(logging.Filter):
    """Inject task_id from contextvar into log records."""

    def filter(self, record):
        record.task_id = current_task_id.get()
        return True

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider


def init_otel() -> Callable:
    """Initialize OTel providers. Returns shutdown function."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return lambda: None

    resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "faster-qwen3-tts"),
        "cloud.region": os.environ.get("GCP_REGION") or "us-central1",
        "k8s.namespace.name": os.environ.get("APP_NAMESPACE") or "app",
        "host.id": os.environ.get("HOSTNAME") or "unknown",
    })

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(insecure=True)))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(insecure=True), export_interval_millis=15000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # Logs
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(insecure=True)))
    set_logger_provider(logger_provider)

    # Bridge stdlib logging -> OTel LogHandler
    handler = LoggingHandler(logger_provider=logger_provider)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addFilter(TaskIdFilter())
    root_logger.addHandler(handler)

    def shutdown():
        tracer_provider.shutdown()
        meter_provider.shutdown()
        logger_provider.shutdown()

    return shutdown

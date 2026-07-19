from prometheus_client import Counter, Histogram

# Instrumentation counters
REQUEST_COUNTER = Counter(
    "prediction_requests_total",
    "Counts the number of prediction requests received.",
    ["mode"],
)

PREDICTION_SAMPLES_COUNTER = Counter(
    "predictions_total",
    "Counts the number of predictions made.",
    ["mode"],
)

OUTCOME_COUNTER = Counter(
    "prediction_outcomes_total",
    "Counts the number of prediction outcomes.",
    ["mode", "prediction"],
)

# Instrumentation histograms
PREDICT_LATENCY_HIST = Histogram(
    "prediction_latency_seconds",
    "Tracks prediction latency in seconds.",
)

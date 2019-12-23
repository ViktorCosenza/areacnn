from models.helpers import Param


def pct_error(pred, label):
    return abs(pred - label) / label


METRICS = [Param("pct_error", pct_error)]


def calculate_metrics(pred, label):
    results = map(lambda metric: metric.param(pred, label), METRICS)
    return list(results)

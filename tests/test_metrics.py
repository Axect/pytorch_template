"""Tests for metrics.py — individual metrics and MetricRegistry."""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from metrics import MSEMetric, MAEMetric, R2Metric, MetricRegistry


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def test_mse_metric():
    """MSEMetric matches torch.nn.functional.mse_loss."""
    torch.manual_seed(42)
    y_pred = torch.randn(50)
    y_true = torch.randn(50)
    metric = MSEMetric()
    expected = torch.nn.functional.mse_loss(y_pred, y_true).item()
    assert abs(metric(y_pred, y_true) - expected) < 1e-6


def test_mae_metric():
    """MAEMetric matches torch.nn.functional.l1_loss."""
    torch.manual_seed(42)
    y_pred = torch.randn(50)
    y_true = torch.randn(50)
    metric = MAEMetric()
    expected = torch.nn.functional.l1_loss(y_pred, y_true).item()
    assert abs(metric(y_pred, y_true) - expected) < 1e-6


def test_r2_metric():
    """R2Metric produces correct value for known input."""
    y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
    y_pred = torch.tensor([2.5, 0.0, 2.0, 8.0])

    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().item()
    expected_r2 = 1 - ss_res / ss_tot

    metric = R2Metric()
    result = metric(y_pred, y_true)
    assert abs(result - expected_r2) < 1e-6


def test_r2_perfect_prediction():
    """R2 = 1.0 when y_pred == y_true."""
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    metric = R2Metric()
    assert metric(y, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------

def test_metric_registry():
    """MetricRegistry computes all registered metrics."""
    torch.manual_seed(42)
    registry = MetricRegistry(["mse", "mae", "r2"])
    y_pred = torch.randn(50)
    y_true = torch.randn(50)
    results = registry.compute(y_pred, y_true)
    assert "mse" in results
    assert "mae" in results
    assert "r2" in results
    assert isinstance(results["mse"], float)
    assert isinstance(results["mae"], float)
    assert isinstance(results["r2"], float)


def test_registry_empty():
    """Empty registry returns empty dict."""
    registry = MetricRegistry()
    y_pred = torch.randn(10)
    y_true = torch.randn(10)
    results = registry.compute(y_pred, y_true)
    assert results == {}

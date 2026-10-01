"""
Threshold rule engine for PSI-based drift alerts.

PSI thresholds from industry convention (credit scoring origin):
  < 0.10  : stable
  0.10-0.20: warning — monitor
  >= 0.20 : critical — investigate
"""
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PSI_WARN = 0.10
PSI_CRIT = 0.20


@dataclass
class Alert:
    window: int
    feature: str
    metric: str
    value: float
    severity: str  # "warning" | "critical"

    def __str__(self):
        return f"[W{self.window:02d}] {self.severity.upper():8s} {self.metric}={self.value:.3f} on '{self.feature}'"


def _severity(value: float, warn: float, crit: float) -> Optional[str]:
    if value >= crit:
        return "critical"
    if value >= warn:
        return "warning"
    return None


def check_alerts(
    metrics: Dict,
    window_id: int,
    psi_warn: float = PSI_WARN,
    psi_crit: float = PSI_CRIT,
) -> List[Alert]:
    alerts: List[Alert] = []

    for feature, psi_val in metrics.get("feature_psi", {}).items():
        if psi_val is None or (isinstance(psi_val, float) and math.isnan(psi_val)):
            continue
        sev = _severity(psi_val, psi_warn, psi_crit)
        if sev:
            a = Alert(window_id, feature, "PSI", psi_val, sev)
            alerts.append(a)
            _log(a)

    for metric_key, label in [("pred_psi", "predictions"), ("error_psi", "errors")]:
        val = metrics.get(metric_key)
        if val is not None and not _is_nan(val):
            sev = _severity(val, psi_warn, psi_crit)
            if sev:
                a = Alert(window_id, label, "PSI", val, sev)
                alerts.append(a)
                _log(a)

    return alerts


def _is_nan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _log(alert: Alert):
    if alert.severity == "critical":
        logging.critical(str(alert))
    else:
        logging.warning(str(alert))


class AlertEngine:
    """Stateful alert accumulator for the full monitoring run."""

    def __init__(self, psi_warn: float = PSI_WARN, psi_crit: float = PSI_CRIT):
        self.psi_warn = psi_warn
        self.psi_crit = psi_crit
        self.history: List[Alert] = []

    def process(self, metrics: Dict, window_id: int) -> List[Alert]:
        new_alerts = check_alerts(metrics, window_id, self.psi_warn, self.psi_crit)
        self.history.extend(new_alerts)
        return new_alerts

    def first_alert_window(self, feature: str, severity: str = "critical") -> Optional[int]:
        for alert in self.history:
            if alert.feature == feature and alert.severity == severity:
                return alert.window
        return None

    def detection_latency(self, injection_window: int, feature: str, severity: str = "critical") -> Optional[int]:
        w = self.first_alert_window(feature, severity)
        return (w - injection_window) if w is not None else None

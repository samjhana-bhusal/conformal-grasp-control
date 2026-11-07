"""
Conformal prediction safety gate for the RL policy.

Design decision log:
  - Gate wraps the RL policy, not the VLM. The VLM provides a confidence score,
    but the gate decides whether to execute based on calibrated risk bounds.
  - Clopper-Pearson exact binomial interval over Bonferroni-corrected candidate
    thresholds. This gives finite-sample, distribution-free guarantees —
    no normality assumption needed.
  - Alpha=0.10 (target 90% success rate among attempted grasps).
    Rationale: 95% is standard in statistics but too conservative for robotics
    where some failure is acceptable. 90% balances safety with throughput.

Theory:
  Given calibration data {(score_i, success_i)}, for each candidate threshold τ:
    - Count successes among trials where score ≥ τ
    - Compute Clopper-Pearson lower bound on true success rate
    - Accept τ if lower_bound ≥ (1 - α)
  Apply Bonferroni correction (δ/K) across K candidate thresholds.
  Choose the smallest valid τ (maximizes coverage).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    threshold: float
    n_calibration: int
    n_candidates_tested: int
    empirical_success_rate: float
    lower_bound: float
    abstention_rate: float
    coverage: float


class ConformalGate:
    """Conformal prediction gate using Clopper-Pearson bounds."""

    def __init__(self, alpha: float = 0.10, delta: float = 0.05, n_candidates: int = 100):
        self.alpha = alpha
        self.delta = delta
        self.n_candidates = n_candidates
        self.threshold = 1.0  # reject everything until calibrated
        self._calibrated = False

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> CalibrationResult:
        """Find the smallest threshold that guarantees (1-α) success rate.

        Args:
            scores: VLM confidence scores, shape (n,)
            labels: binary success labels, shape (n,)
        """
        candidates = np.linspace(0.0, 1.0, self.n_candidates)
        corrected_delta = self.delta / self.n_candidates  # Bonferroni

        best_tau = 1.0
        best_lb = 0.0
        best_sr = 0.0

        for tau in candidates:
            mask = scores >= tau
            n_above = mask.sum()
            if n_above < 5:  # too few samples
                continue

            n_success = labels[mask].sum()
            sr = n_success / n_above

            # Clopper-Pearson lower bound
            if n_success == 0:
                lb = 0.0
            else:
                lb = stats.beta.ppf(corrected_delta, n_success, n_above - n_success + 1)

            if lb >= (1.0 - self.alpha):
                if tau < best_tau:
                    best_tau = tau
                    best_lb = lb
                    best_sr = sr

        self.threshold = best_tau
        self._calibrated = True

        mask = scores >= best_tau
        result = CalibrationResult(
            threshold=best_tau,
            n_calibration=len(scores),
            n_candidates_tested=self.n_candidates,
            empirical_success_rate=best_sr,
            lower_bound=best_lb,
            abstention_rate=1.0 - mask.mean(),
            coverage=mask.mean(),
        )
        return result

    def should_attempt(self, score: float) -> bool:
        """Return True if the VLM score passes the safety gate."""
        return score >= self.threshold

    def gate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Return boolean mask of which grasps to attempt."""
        return scores >= self.threshold

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

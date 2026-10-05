"""Static exposure policies."""

from __future__ import annotations

import random
from dataclasses import dataclass


def _unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(slots=True)
class PolicyDecision:
    keep_probability: float
    shown: bool


class StaticCOSREFPolicy:
    """Operationalize COSREF parameters as exposure keep probabilities.

    This implementation does not claim that the resulting probability is
    mathematically identical to an effective spreading rate in prior work.
    """

    def __init__(
        self,
        omega_intra: float,
        omega_inter: float,
        seed: int,
    ) -> None:
        self.omega_intra = _unit_interval("omega_intra", omega_intra)
        self.omega_inter = _unit_interval("omega_inter", omega_inter)
        self._seed = int(seed)

    def update(self, omega_intra: float, omega_inter: float) -> None:
        self.omega_intra = _unit_interval("omega_intra", omega_intra)
        self.omega_inter = _unit_interval("omega_inter", omega_inter)

    def keep_probability(
        self,
        user_community: str,
        author_community: str,
        risk_score: float,
    ) -> float:
        risk = _unit_interval("risk_score", risk_score)
        omega = (
            self.omega_intra
            if user_community == author_community
            else self.omega_inter
        )
        return 1.0 - risk * (1.0 - omega)

    def decide(
        self,
        user_community: str,
        author_community: str,
        risk_score: float,
        decision_key: int,
    ) -> PolicyDecision:
        probability = self.keep_probability(
            user_community, author_community, risk_score
        )
        draw = random.Random(hash((self._seed, decision_key))).random()
        return PolicyDecision(probability, draw < probability)


class NoInterventionPolicy(StaticCOSREFPolicy):
    def __init__(self, seed: int) -> None:
        super().__init__(1.0, 1.0, seed)


class GlobalThrottlePolicy(StaticCOSREFPolicy):
    def __init__(self, keep_probability: float, seed: int) -> None:
        super().__init__(keep_probability, keep_probability, seed)

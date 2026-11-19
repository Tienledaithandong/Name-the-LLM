"""Historical transductive reporting must not bypass the current protocol."""

from __future__ import annotations

import sys

import pytest

from experiments import stress_k


@pytest.mark.parametrize("extra", [[], ["--reserve-fresh-holdout"]])
def test_historical_stress_cli_refuses_to_report_or_reserve(monkeypatch, extra: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["stress_k.py", *extra])
    with pytest.raises(SystemExit, match="transductive methods are prohibited"):
        stress_k.main()

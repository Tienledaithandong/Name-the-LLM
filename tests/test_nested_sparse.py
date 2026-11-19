from __future__ import annotations

import numpy as np

from experiments.run_nested_sparse import choose_char_weight


def test_inner_weight_selection_prefers_better_inner_predictions() -> None:
    y = np.array([0, 1, 0, 1])
    char = np.array([[3, 0], [0, 3], [3, 0], [0, 3]])
    style = np.array([[0, 3], [3, 0], [0, 3], [3, 0]])
    weight, scores = choose_char_weight(y, char, style)
    assert weight == 1.0
    assert scores["1.0"] == 1.0
    assert scores["0.0"] == 0.0

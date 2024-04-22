import numpy as np

from thuban.util import remove_pairless_points


def test_remove_pairless_points_no_pairs():
    """makes sure empty lists are returned when no pairs are present"""
    a = np.zeros((100, 2))
    b = np.ones((50, 2)) * 10
    out_a, out_b = remove_pairless_points(a, b, max_distance=2)
    assert len(out_a) == 0
    assert len(out_b) == 0
    assert len(a) == 100
    assert len(b) == 50


def test_remove_pairless_points_all_pairs():
    """makes sure all points are returned when all are close"""
    a = np.zeros((100, 2))
    b = np.ones((50, 2))
    out_a, out_b = remove_pairless_points(a, b, max_distance=2)
    assert len(out_a) == 50
    assert len(out_b) == 50
    assert len(a) == 100
    assert len(b) == 50

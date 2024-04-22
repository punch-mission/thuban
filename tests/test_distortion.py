import numpy as np
from astropy.wcs import DistortionLookupTable

from thuban.distortion import (compute_distortion, filter_distortion_table,
                               make_empty_distortion_model)


def test_filter_distortion_table_nan_edges():
    """from https://github.com/svank/wispr_analysis/blob/main/wispr_analysis/tests/test_image_alignment.py"""
    data = np.zeros((100, 100))

    # Put nans at the edges that should be cleared, and values to fill them in
    # with
    data[2] = 9
    data[-6] = 2
    data[:, 3] = 5
    data[:, -11] = -12.2

    data[:2] = np.nan
    data[-5:] = np.nan
    data[:, :3] = np.nan
    data[:, -10:] = np.nan

    filtered = filter_distortion_table(data, blur_sigma=0, med_filter_size=0)

    # The trimmed edges should be replaced with the edge values
    np.testing.assert_equal(filtered[:2, 4:-11], 9)
    np.testing.assert_equal(filtered[-5:, 4:-11], 2)
    np.testing.assert_equal(filtered[:, :3], 5)
    np.testing.assert_equal(filtered[:, -10:], -12.2)


def test_filter_distortion_table_nans_center():
    """from https://github.com/svank/wispr_analysis/blob/main/wispr_analysis/tests/test_image_alignment.py"""
    data = np.zeros((100, 100))

    # Put nans elsewhere that should not be trimmed
    # And surround them with values to be medianed
    data[2, :50] = np.nan
    data[1, :50] = 1
    data[3, :50] = 3

    data[50:, -11] = np.nan
    data[50:, -10] = 1.25
    data[50:, -12] = 1.75

    data[29, 29:32] = 1
    data[30, 29:32] = 0
    data[31, 29:32] = 10
    data[30, 30] = np.nan

    data[60:70, 60:70] = 10
    data[63:67, 63:67] = np.nan

    filtered = filter_distortion_table(data, blur_sigma=0, med_filter_size=0)

    # The nans should have been replaced with a neighborhood median
    np.testing.assert_equal(filtered[2, :49], np.median([1, 3]))
    np.testing.assert_equal(filtered[2, 49], np.median([0, 0, 0, 1, 1, 3, 3]))

    np.testing.assert_equal(filtered[50, -11], np.median([0, 0, 0, 1.25, 1.25, 1.75, 1.75]))
    np.testing.assert_equal(filtered[51:, -11], np.median([1.25, 1.75]))

    np.testing.assert_equal(filtered[30, 30], np.median([1, 1, 1, 0, 0, 10, 10, 10]))

    np.testing.assert_equal(filtered[63:67, 63:67], 10)


def test_filter_distortion_table_median_filter_image():
    """from https://github.com/svank/wispr_analysis/blob/main/wispr_analysis/tests/test_image_alignment.py"""
    data = np.zeros((50, 50))

    data[30, 40] = 5

    filtered = filter_distortion_table(data, blur_sigma=0)

    assert filtered[30, 40] == 0


def test_filter_distortion_table_gaussian_filter_image():
    """from https://github.com/svank/wispr_analysis/blob/main/wispr_analysis/tests/test_image_alignment.py"""
    data = np.zeros((50, 50))

    data[30, 40] = 5

    filtered = filter_distortion_table(data, med_filter_size=0)

    assert filtered[30, 40] > 0
    assert filtered[30, 40] < 5


def test_make_empty_distortion_model():
    """tests making an empty distortion model"""
    data = np.zeros((100, 100))
    num_bins = 10
    cpdis1, cpdis2 = make_empty_distortion_model(num_bins, data)
    assert isinstance(cpdis1, DistortionLookupTable)
    assert isinstance(cpdis2, DistortionLookupTable)
    assert cpdis1.data.shape == (10, 10)
    assert cpdis2.data.shape == (10, 10)
    assert np.all(cpdis1.data == 0)
    assert np.all(cpdis2.data == 0)


def test_compute_distortion_random_arrays():
    """given random arrays computes the distortion model"""
    num_bins = 75
    x = np.random.randint(0, 1023, (10000, 2))
    y = np.random.randint(0, 1023, (10000, 2))

    cpdis1, cpdis2 = compute_distortion((1024, 1024), x, y, num_bins=num_bins)
    assert isinstance(cpdis1, DistortionLookupTable)
    assert isinstance(cpdis2, DistortionLookupTable)
    assert cpdis1.data.shape == (num_bins, num_bins)
    assert cpdis2.data.shape == (num_bins, num_bins)

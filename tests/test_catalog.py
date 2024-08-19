import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from thuban.catalog import (filter_for_visible_stars, find_catalog_in_image,
                            load_hipparcos_catalog, load_raw_hipparcos_catalog)


def test_catalog_loads():
    """makes sure the full catalog is loaded"""
    catalog = load_hipparcos_catalog()
    assert isinstance(catalog, pd.DataFrame)
    assert len(catalog) == 113759  # ensure we get the full catalog


def test_catalog_filters():
    """tests that the catalog gets filtered by the requested magnitude"""
    catalog = load_hipparcos_catalog()
    filtered = filter_for_visible_stars(catalog, dimmest_magnitude=6)
    assert np.all(filtered["Vmag"] <= 6)


def test_find_catalog_in_image():
    """tests that coordinates are properly selected"""
    catalog = load_hipparcos_catalog()
    w = WCS(naxis=2)
    w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.set_pv([(2, 1, 45.0)])
    reduced_catalog = find_catalog_in_image(catalog, w, image_shape=(2048, 1024))
    assert len(reduced_catalog) == 13_459
    assert np.all(reduced_catalog["x_pix"] >= 0)
    assert np.all(reduced_catalog["x_pix"] < 2048)
    assert np.all(reduced_catalog["y_pix"] >= 0)
    assert np.all(reduced_catalog["y_pix"] < 1024)


@pytest.mark.slow
def test_download_catalog():
    """tests that the full catalog downloads"""
    raw_hipparcos = load_raw_hipparcos_catalog()
    reduced_hipparcos = filter_for_visible_stars(raw_hipparcos, dimmest_magnitude=14)
    assert len(reduced_hipparcos) == 113759  # ensure we get the full catalog

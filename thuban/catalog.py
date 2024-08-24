import os
from typing import Callable

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, NoConvergence

__all__ = [
    "HIPPARCOS_URL",
    "load_hipparcos_catalog",
    "load_raw_hipparcos_catalog",
    "filter_for_visible_stars",
    "find_catalog_in_image",
]

_ROOT = os.path.abspath(os.path.dirname(__file__))
HIPPARCOS_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"


def get_data(path):
    return os.path.join(_ROOT, "data", path)


def load_hipparcos_catalog(catalog_path: str = get_data("reduced_hip.csv")) -> pd.DataFrame:
    """Load the Hipparcos catalog from the local, reduced version. This version only keeps necessary columns.

    Parameters
    ----------
    catalog_path : str
        path to the catalog, defaults to a provided version

    Returns
    -------
    pd.DataFrame
        loaded catalog with selected columns
    """
    return pd.read_csv(catalog_path)


def load_raw_hipparcos_catalog(catalog_path: str = HIPPARCOS_URL) -> pd.DataFrame:
    """Download hipparcos catalog from website. Not recommended for routine use.

    Parameters
    ----------
    catalog_path : str
        path to the Hipparcos catalog

    Returns
    -------
    pd.DataFrame
        loaded catalog with selected columns
    """
    column_names = (
        "Catalog",
        "HIP",
        "Proxy",
        "RAhms",
        "DEdms",
        "Vmag",
        "VarFlag",
        "r_Vmag",
        "RAdeg",
        "DEdeg",
        "AstroRef",
        "Plx",
        "pmRA",
        "pmDE",
        "e_RAdeg",
        "e_DEdeg",
        "e_Plx",
        "e_pmRA",
        "e_pmDE",
        "DE:RA",
        "Plx:RA",
        "Plx:DE",
        "pmRA:RA",
        "pmRA:DE",
        "pmRA:Plx",
        "pmDE:RA",
        "pmDE:DE",
        "pmDE:Plx",
        "pmDE:pmRA",
        "F1",
        "F2",
        "---",
        "BTmag",
        "e_BTmag",
        "VTmag",
        "e_VTmag",
        "m_BTmag",
        "B-V",
        "e_B-V",
        "r_B-V",
        "V-I",
        "e_V-I",
        "r_V-I",
        "CombMag",
        "Hpmag",
        "e_Hpmag",
        "Hpscat",
        "o_Hpmag",
        "m_Hpmag",
        "Hpmax",
        "HPmin",
        "Period",
        "HvarType",
        "moreVar",
        "morePhoto",
        "CCDM",
        "n_CCDM",
        "Nsys",
        "Ncomp",
        "MultFlag",
        "Source",
        "Qual",
        "m_HIP",
        "theta",
        "rho",
        "e_rho",
        "dHp",
        "e_dHp",
        "Survey",
        "Chart",
        "Notes",
        "HD",
        "BD",
        "CoD",
        "CPD",
        "(V-I)red",
        "SpType",
        "r_SpType",
    )
    df = pd.read_csv(
        catalog_path,
        sep="|",
        names=column_names,
        usecols=["HIP", "Vmag", "RAdeg", "DEdeg", "Plx"],
        na_values=["     ", "       ", "        ", "            "],
    )
    df["distance"] = 1000 / df["Plx"]
    df = df[df["distance"] > 0]
    return df.iloc[np.argsort(df["Vmag"])]


def filter_for_visible_stars(catalog: pd.DataFrame, dimmest_magnitude: float = 6) -> pd.DataFrame:
    """Filters to only include stars brighter than a given magnitude

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog loaded from `~thuban.catalog.load_hipparcos_catalog` or `~thuban.catalog.load_raw_hipparcos_catalog`

    dimmest_magnitude : float
        the dimmest magnitude to keep

    Returns
    -------
    pd.DataFrame
        a catalog with stars dimmer than the `dimmest_magnitude` removed
    """
    return catalog[catalog["Vmag"] < dimmest_magnitude]


def find_catalog_in_image(
    catalog: pd.DataFrame, wcs: WCS, image_shape: (int, int), mask: Callable = None,
        mode: str = "all"
) -> np.ndarray:
    """Using the provided WCS converts the RA/DEC catalog into pixel coordinates

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog loaded from `~thuban.catalog.load_hipparcos_catalog` or `~thuban.catalog.load_raw_hipparcos_catalog`
    wcs : WCS
        the world coordinate system of a given image
    image_shape: (int, int)
        the shape of the image array associated with the WCS, used to only consider stars with coordinates in image
    mode : str
        either "all" or "wcs",
        see
        <https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.to_pixel>

    Returns
    -------
    np.ndarray
        pixel coordinates of stars in catalog that are present in the image
    """
    try:
        xs, ys = SkyCoord(
            ra=np.array(catalog["RAdeg"]) * u.degree,
            dec=np.array(catalog["DEdeg"]) * u.degree,
            distance=np.array(catalog["distance"]) * u.parsec,
        ).to_pixel(wcs, mode=mode)
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    bounds_mask = (0 <= xs) * (xs < image_shape[0]) * (0 <= ys) * (ys < image_shape[1])

    if mask is not None:
        bounds_mask *= mask(xs, ys)

    reduced_catalog = catalog[bounds_mask].copy()
    reduced_catalog["x_pix"] = xs[bounds_mask]
    reduced_catalog['y_pix'] = ys[bounds_mask]
    return reduced_catalog

import numpy as np
from astropy.wcs import WCS
from sklearn.neighbors import KDTree


def remove_pairless_points(a: np.ndarray, b: np.ndarray, max_distance=20) -> (np.ndarray, np.ndarray):
    """Given two lists (a and b), remove points that don't have a neighbor within `max_distance` in the other set

    Parameters
    ----------
    a : np.ndarray
        an array of points [N, d] where N is the number of points and d is the dimensionality
    b : np.ndarray
        an array of points [N, d] where N is the number of points and d is the dimensionality
    max_distance : float
        points without a neighbor in the other set within `max_distance` will be removed

    Returns
    -------
    (np.ndarray, np.ndarray)
        point sets with pairless points removed
    """
    tree = KDTree(a)
    distances, indices = tree.query(b)
    b = b[distances.squeeze() <= max_distance]

    if len(b) == 0:
        return np.array([]), np.array([])

    tree = KDTree(b)
    distances, indices = tree.query(a)
    a = a[distances.squeeze() <= max_distance]

    if len(a) == 0:
        return np.array([]), np.array([])

    tree = KDTree(a)
    distances, indices = tree.query(b)
    a = a[indices].squeeze()
    return a, b


def find_celestial_wcs(hdul):
    # modified from https://github.com/svank/remove_starfield/blob/main/remove_starfield/utils.py
    # If the FITS file is compressed, the first HDU has no data. Search for the
    # first non-empty hdu
    hdu_num = 0
    while hdul[hdu_num].data is None:
        hdu_num += 1

    hdr = hdul[hdu_num].header
    # Search for a celestial WCS
    for key in ' ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        found_wcs = WCS(hdr, hdul, key=key)
        ctypes = sorted(found_wcs.wcs.ctype)
        ctypes = [c[:3] for c in ctypes]
        if 'DEC' in ctypes and 'RA-' in ctypes:
            return hdu_num, key

    raise ValueError("No celestial WCS found")


def sparsify(coords: np.ndarray, radius: float) -> np.ndarray:
    """
    Returns a sparsified version of the input coordinates array,
    where only the points that are at least `radius` distance
    apart from each other are kept.

    Parameters:
    -----------
    coords : np.ndarray
        The input array of shape (n, 2) containing the coordinates of the points.
    radius : float
        The minimum distance between two points to be kept in the output array.

    Returns:
    --------
    np.ndarray
        The sparsified array of shape (m, 2),
        where `m` is the number of points that are at least `radius` distance apart
        from each other.
    """
    _coords = coords.copy()
    deleted_coords = np.zeros([], dtype=int)
    sparse_coords = []

    for i, s in enumerate(_coords):
        if i not in deleted_coords:
            distances = np.linalg.norm(_coords - s, axis=1)
            idxs = np.flatnonzero(distances < radius)
            sparse_coords.append(s)
            deleted_coords = np.hstack([deleted_coords, idxs])

    return np.array(sparse_coords)

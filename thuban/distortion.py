import warnings

import numpy as np
import scipy
import scipy.stats
from astropy.wcs import DistortionLookupTable


def filter_distortion_table(data: np.ndarray, blur_sigma: float = 4, med_filter_size: float = 3) -> np.ndarray:
    """
    Returns a filtered copy of a distortion map table.

    Any rows/columns at the edges that are all NaNs will be removed and
    replaced with a copy of the closest non-removed edge at the end of
    processing.

    Any NaN values that don't form a complete edge row/column will be replaced
    with the median of all surrounding non-NaN pixels.

    Then median filtering is performed across the whole map to remove outliers,
    and Gaussian filtering is applied to accept only slowly-varying
    distortions.

    Parameters
    ----------
    data
        The distortion map to be filtered
    blur_sigma : float
        The number of pixels constituting one standard deviation of the
        Gaussian kernel. Set to 0 to disable Gaussian blurring.
    med_filter_size : int
        The size of the local neighborhood to consider for median filtering.
        Set to 0 to disable median filtering.

    Notes
    -----
    Modified from https://github.com/svank/wispr_analysis/blob/main/wispr_analysis/image_alignment.py
    """
    data = data.copy()

    # Trim empty (all-nan) rows and columns
    trimmed = []
    i = 0
    while np.all(np.isnan(data[0])):
        i += 1
        data = data[1:]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[-1])):
        i += 1
        data = data[:-1]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[:, 0])):
        i += 1
        data = data[:, 1:]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[:, -1])):
        i += 1
        data = data[:, :-1]
    trimmed.append(i)

    # Replace interior nan values with the median of the surrounding values.
    # We're filling in from neighboring pixels, so if there are any nan pixels
    # fully surrounded by nan pixels, we need to iterate a few times.
    while np.any(np.isnan(data)):
        nans = np.nonzero(np.isnan(data))
        replacements = np.zeros_like(data)
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice")
            for r, c in zip(*nans):
                r1, r2 = r - 1, r + 2
                c1, c2 = c - 1, c + 2
                r1, r2 = max(r1, 0), min(r2, data.shape[0])
                c1, c2 = max(c1, 0), min(c2, data.shape[1])

                replacements[r, c] = np.nanmedian(data[r1:r2, c1:c2])
        data[nans] = replacements[nans]

    # Median-filter the whole image
    if med_filter_size:
        data = scipy.ndimage.median_filter(data, size=med_filter_size, mode="reflect")

    # Gaussian-blur the whole image
    if blur_sigma > 0:
        data = scipy.ndimage.gaussian_filter(data, sigma=blur_sigma)

    # Replicate the edge rows/columns to replace those we trimmed earlier
    return np.pad(data, [trimmed[0:2], trimmed[2:]], mode="edge")


def make_empty_distortion_model(num_bins: int, image: np.ndarray) -> (DistortionLookupTable, DistortionLookupTable):
    """ Create an empty distortion table

    Parameters
    ----------
    num_bins : int
        number of histogram bins in the distortion model, i.e. the size of the distortion model is (num_bins, num_bins)
    image : np.ndarray
        image to create a distortion model for

    Returns
    -------
    (DistortionLookupTable, DistortionLookupTable)
        x and y distortion models
    """
    # make an initial empty distortion model
    r = np.linspace(0, image.shape[0], num_bins + 1)
    c = np.linspace(0, image.shape[1], num_bins + 1)
    r = (r[1:] + r[:-1]) / 2
    c = (c[1:] + c[:-1]) / 2

    err_px, err_py = r, c
    err_x = np.zeros((num_bins, num_bins))
    err_y = np.zeros((num_bins, num_bins))

    cpdis1 = DistortionLookupTable(
        -err_x.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )
    cpdis2 = DistortionLookupTable(
        -err_y.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )
    return cpdis1, cpdis2


def compute_distortion(
    img_shape: (int, int),
        catalog_positions: np.ndarray,
        found_positions: np.ndarray,
        distortion_limit: float =20, num_bins: int =75, blur_sigma: float =4, med_filter_size: int=3
) -> (DistortionLookupTable, DistortionLookupTable):
    """ Given the derived catalog and actual star positions, determines the distortion

    Parameters
    ----------
    img_shape : (int, int)
        shape of the image to compute the distortion for
    catalog_positions : np.ndarray
        pixel positions of the stars according to the catalog; (N, 2) in shape where N is number of stars
    found_positions : np.ndarray
        pixel positions of the stars found by sep in the image; (N, 2) in shape where N is the number of stars
    distortion_limit : float
        stars with distortion greater than this are not considered as they're a mismatch instead of true distortion
    num_bins : int
        number of histogram bins in the distortion model, i.e. the size of the distortion model is (num_bins, num_bins)
    blur_sigma : float
        The number of pixels constituting one standard deviation of the
        Gaussian kernel. Set to 0 to disable Gaussian blurring.
    med_filter_size : int
        The size of the local neighborhood to consider for median filtering.
        Set to 0 to disable median filtering.

    Returns
    -------
    (DistortionLookupTable, DistortionLookupTable)
        x and y distortion models determined from the star locations
    """
    distortion = catalog_positions - found_positions
    indices_to_keep = np.where(
        (np.abs(distortion[:, 0]) < distortion_limit) * (np.abs(distortion[:, 1]) < distortion_limit)
    )[0]
    distortion = distortion[indices_to_keep, :]
    found_positions = found_positions[indices_to_keep, :]

    xbins, r, c, _ = scipy.stats.binned_statistic_2d(
        found_positions[:, 1],
        found_positions[:, 0],
        distortion[:, 0],
        "median",
        (num_bins, num_bins),
        expand_binnumbers=True,
        range=((0, img_shape[1]), (0, img_shape[0])),
    )

    ybins, _, _, _ = scipy.stats.binned_statistic_2d(
        found_positions[:, 1],
        found_positions[:, 0],
        distortion[:, 1],
        "median",
        (num_bins, num_bins),
        expand_binnumbers=True,
        range=((0, img_shape[1]), (0, img_shape[0])),
    )

    r = (r[1:] + r[:-1]) / 2
    c = (c[1:] + c[:-1]) / 2

    xbins = filter_distortion_table(xbins, blur_sigma=blur_sigma, med_filter_size=med_filter_size)
    ybins = filter_distortion_table(ybins, blur_sigma=blur_sigma, med_filter_size=med_filter_size)

    err_px, err_py = r, c
    err_x = xbins
    err_y = ybins

    cpdis1 = DistortionLookupTable(
        -err_x.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )
    cpdis2 = DistortionLookupTable(
        -err_y.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )

    return cpdis1, cpdis2

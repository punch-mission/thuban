import warnings

import numpy as np
import pandas as pd
import sep
from astropy.wcs import WCS, utils
from lmfit import Parameters, minimize

from thuban.catalog import (filter_for_visible_stars, find_catalog_in_image,
                            load_hipparcos_catalog)
from thuban.error import ConvergenceWarning
from thuban.util import remove_pairless_points


def convert_cd_matrix_to_pc_matrix(wcs):
    if hasattr(wcs.wcs, 'cd'):
        cdelt1, cdelt2 = utils.proj_plane_pixel_scales(wcs)
        crota = np.arccos(wcs.wcs.cd[0, 0]/cdelt1)
        print("CROTA", np.rad2deg(crota))
        new_wcs = WCS(naxis=2)
        new_wcs.wcs.ctype = wcs.wcs.ctype
        new_wcs.wcs.crval = wcs.wcs.crval
        new_wcs.wcs.crpix = wcs.wcs.crpix
        new_wcs.wcs.pc = calculate_pc_matrix(crota, (cdelt1, cdelt2))
        new_wcs.wcs.cdelt = (-cdelt1, cdelt2)
        new_wcs.wcs.cunit = 'deg', 'deg'
        return new_wcs
    else:  # noqa RET505
        return wcs


def calculate_pc_matrix(crota: float, cdelt: (float, float)) -> np.ndarray:
    """Calculate a PC matrix given CROTA and CDELT

    Parameters
    ----------
    crota : float
        rotation angle from the WCS
    cdelt : float
        pixel size from the WCS

    Returns
    -------
    np.ndarray
        PC matrix
    """
    return np.array(
        [
            [np.cos(crota), np.sin(crota) * (cdelt[1] / cdelt[0])],
            [-np.sin(crota) * (cdelt[0] / cdelt[1]), np.cos(crota)],
        ]
    )


def _residual(params: Parameters,
              observed_coords: np.ndarray, catalog: pd.DataFrame, guess_wcs: WCS, ii,
              image_shape: (int, int), max_distance=15):
    """Residual used when optimizing the pointing

    Parameters
    ----------
    params : Parameters
        optimization parameters from lmfit
    observed_coords : np.ndarray
        pixel coordinates of stars observed in the image, i.e. the coordinates found by sep of the actual star location
    catalog : pd.DataFrame
        image catalog of stars to match against
    guess_wcs : WCS
        initial guess of the world coordinate system, must overlap with the true WCS
    x_lim : (int, int)
        start and end x pixel coordinates to use of the image for pointing determination
    y_lim : (int, int)
        start and end y pixel coordinates to use of the image for the pointing determination
    n : int
        number of stars to randomly select when doing each iteration of the optimization

    Returns
    -------
    np.ndarray
        residual
    """
    refined_wcs = WCS(naxis=2)
    refined_wcs.wcs.ctype = guess_wcs.wcs.ctype
    refined_wcs.wcs.cunit = guess_wcs.wcs.cunit
    refined_wcs.wcs.cdelt = guess_wcs.wcs.cdelt
    refined_wcs.wcs.crpix = guess_wcs.wcs.crpix
    refined_wcs.wcs.crval = (params["crval1"].value, params["crval2"].value)
    refined_wcs.wcs.pc = calculate_pc_matrix(params["crota"], refined_wcs.wcs.cdelt)
    refined_wcs.cpdis1 = guess_wcs.cpdis1   # TODO what if there is no distortion?
    refined_wcs.cpdis2 = guess_wcs.cpdis2

    reduced_catalog = find_catalog_in_image(catalog, refined_wcs, image_shape=image_shape)
    refined_coords = np.stack([reduced_catalog['x_pix'], reduced_catalog['y_pix']], axis=-1)

    return [np.min(np.linalg.norm(observed_coords - coord, axis=-1)) for coord in refined_coords[ii]]

    # if not np.isinf(max_distance):
    #     observed_coords, refined_coords = remove_pairless_points(observed_coords, refined_coords, max_distance)
    # ii = np.min([np.array(list(range(n))),
    #              np.random.random_sample(np.arange(np.min(observed_coords.shape[0] - 1, n), )], axis=0).astype(int)

    # return observed_coords[ii] - refined_coords[ii]
    # return observed_coords - refined_coords


def refine_pointing(image, guess_wcs, observed_coords=None, catalog=None,
                    background_width=16, background_height=16, max_distance=15, top_stars=100,
                    detection_threshold=5, num_stars=30, max_trials=15, chisqr_threshold=0.1,
                    dimmest_magnitude=6.0, method='leastsq'):
    """ Refine the pointing for an image

    Parameters
    ----------
    image : np.ndarray
        the brightnesses of the image, no preprocessing necessary
    guess_wcs : WCS
        initial guess for th world coordinate system, must overlap with the true WCS
    file_index
    observed_coords
    catalog
    background_width
    background_height
    detection_threshold
    x_lim
    y_lim
    n

    Returns
    -------

    """
    if catalog is None:
        catalog = filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=dimmest_magnitude)

    if observed_coords is None:
        # find the stars in the background removed image
        background = sep.Background(image, bw=background_width, bh=background_height)
        data_sub = image - background
        objects = sep.extract(data_sub, detection_threshold, err=background.globalrms)
        observed_coords = np.stack([objects["x"], objects["y"]], axis=-1)

    # TODO: make masking and filtering better!
    # bad_indices = np.where(
    #     (object_positions[:, 0] > 800) * (object_positions[:, 0] < 1300) * (object_positions[:, 1] > 800) * (
    #                 object_positions[:, 1] < 1300))[0]
    # removed = set(range(object_positions.shape[0])) - set(bad_indices)
    # object_positions = object_positions[np.array(list(removed)).astype(int)]

    # set up the optimization
    params = Parameters()
    params.add("crota", value=np.arctan2(guess_wcs.wcs.pc[1, 0], guess_wcs.wcs.pc[0, 0]), min=0, max=2 * np.pi)
    params.add("crval1", value=guess_wcs.wcs.crval[0], min=-180, max=180, vary=True)
    params.add("crval2", value=guess_wcs.wcs.crval[1], min=-90, max=90, vary=True)

    # indices = np.arange(np.min([top_stars, observed_coords.shape[0]]))
    # ii = random.choices(indices, k=num_stars)
    ii = np.arange(num_stars)

    # optimize
    trial_num = 0
    result_wcses, result_minimizations = [], []
    while trial_num < max_trials:
        try:
            out = minimize(_residual, params, method=method,
                           args=(observed_coords, catalog, guess_wcs, ii, image.shape, max_distance))
            chisqr = out.chisqr
            result_minimizations.append(out)
        except IndexError:
            result_wcs = guess_wcs
            chisqr = np.inf
            warnings.warn("guess_wcs returned because pointing minimization did not converge properly.",
                          ConvergenceWarning)
        else:
            # construct the result
            result_wcs = WCS(naxis=2)
            result_wcs.wcs.ctype = guess_wcs.wcs.ctype
            result_wcs.wcs.cunit = guess_wcs.wcs.cunit
            result_wcs.wcs.cdelt = guess_wcs.wcs.cdelt
            result_wcs.wcs.crpix = guess_wcs.wcs.crpix
            result_wcs.wcs.crval = (out.params["crval1"].value, out.params["crval2"].value)
            result_wcs.wcs.pc = calculate_pc_matrix(out.params["crota"], result_wcs.wcs.cdelt)
            result_wcs.cpdis1 = guess_wcs.cpdis1  # TODO: what if there is no known distortion
            result_wcs.cpdis2 = guess_wcs.cpdis2
        result_wcses.append(result_wcs)
        trial_num += 1
        if chisqr < chisqr_threshold:
            break

    chisqrs = [s.chisqr for s in result_minimizations]
    best_index = np.argmin(chisqrs)

    return result_wcses[best_index], observed_coords, result_minimizations[best_index], trial_num


def get_star_lists(catalog, refined_wcs, observed_coords, max_distance=20) -> (np.ndarray, np.ndarray, np.ndarray):
    """Calculates the stars vi

    Parameters
    ----------
    catalog
    refined_wcs
    observed_coords
    max_distance

    Returns
    -------

    """
    refined_coords = find_catalog_in_image(catalog, refined_wcs, mode='all')
    no_distortion = find_catalog_in_image(catalog, refined_wcs, mode='wcs')

    refined_coords, observed_coords = remove_pairless_points(refined_coords, observed_coords, max_distance=max_distance)
    observed_coords, no_distortion = remove_pairless_points(observed_coords, no_distortion, max_distance=max_distance)

    return observed_coords.squeeze(), refined_coords.squeeze(), no_distortion.squeeze()
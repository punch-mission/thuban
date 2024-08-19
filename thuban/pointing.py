import warnings

import numpy as np
import pandas as pd
import sep_pjw as sep
from astropy.wcs import WCS, utils
from lmfit import Parameters, minimize

from thuban.catalog import (filter_for_visible_stars, find_catalog_in_image,
                            load_hipparcos_catalog)
from thuban.error import ConvergenceWarning, RepeatedStarWarning
from thuban.util import remove_pairless_points


def convert_cd_matrix_to_pc_matrix(wcs):
    if hasattr(wcs.wcs, 'cd'):
        cdelt1, cdelt2 = utils.proj_plane_pixel_scales(wcs)
        crota = np.arccos(wcs.wcs.cd[0, 0]/cdelt1)
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
    """
    Calculate a PC matrix given CROTA and CDELT.

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
            [np.cos(crota), np.sin(crota) * (cdelt[0] / cdelt[1])],
            [-np.sin(crota) * (cdelt[1] / cdelt[0]), np.cos(crota)],
        ],
    )



def _residual(params: Parameters,
              observed_coords: np.ndarray,
              catalog: pd.DataFrame,
              guess_wcs: WCS,
              n: int,
              image_shape: (int, int)):
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
    n: int
        number of stars to use in calculating the error
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

    edge = 300
    image_bounds = (image_shape[0] - edge, image_shape[1] - edge)
    refined_coords = np.array([c for c in refined_coords
                      if (c[0] > edge) and (c[1] > edge) and (c[0] < image_bounds[0]) and (c[1] < image_bounds[1])])

    # repeat coordinates if there aren't enough stars to meet the n threshold and issue a warning
    if len(refined_coords) < n:
        refined_coords = np.concatenate([refined_coords, refined_coords[:(len(refined_coords) - n)]])
        warnings.warn("Stars repeated in solving because too few found. "
                      "Try decreasing `num_stars` or increasing `dimmest_magnitude`.",
                      RepeatedStarWarning)

    sigma = 3
    out = np.array([np.min(np.linalg.norm(observed_coords - coord, axis=-1)) for coord in refined_coords[:n]])
    mean, stdev = np.mean(out), np.std(out)
    out[out > mean + sigma * stdev] = 0.0
    return out


def refine_pointing_wrapper(image, guess_wcs, file_num, observed_coords=None, catalog=None,
                    background_width=16, background_height=16,
                    detection_threshold=5, num_stars=30, max_trials=15, chisqr_threshold=0.1,
                    dimmest_magnitude=6.0, method='leastsq'):
    new_wcs, observed_coords, solution, trial_num = refine_pointing(image,
                                                                    guess_wcs,
                                                                    observed_coords=observed_coords, catalog=catalog,
                    background_width=background_width, background_height=background_height,
                    detection_threshold=detection_threshold, num_stars=num_stars, max_trials=max_trials,
                            chisqr_threshold=chisqr_threshold,
                    dimmest_magnitude=dimmest_magnitude, method=method)
    return new_wcs, observed_coords, solution, trial_num, file_num


def refine_pointing(image, guess_wcs, observed_coords=None, catalog=None,
                    background_width=16, background_height=16,
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
        objects = pd.DataFrame(objects).sort_values('flux')[-3*num_stars:]
        observed_coords = np.stack([objects["x"], objects["y"]], axis=-1)

    # set up the optimization
    params = Parameters()
    delta_ratio = guess_wcs.wcs.cdelt[0] / guess_wcs.wcs.cdelt[1]
    initial_crota = np.arctan2(guess_wcs.wcs.pc[0, 1]/delta_ratio, guess_wcs.wcs.pc[0, 0])
    params.add("crota", value=initial_crota, min=-np.pi, max=np.pi)
    params.add("crval1", value=guess_wcs.wcs.crval[0], min=-180, max=180, vary=True)
    params.add("crval2", value=guess_wcs.wcs.crval[1], min=-90, max=90, vary=True)

    # optimize
    trial_num = 0
    result_wcses, result_minimizations = [], []
    while trial_num < max_trials:
        try:
            out = minimize(_residual, params, method=method,
                           args=(observed_coords, catalog, guess_wcs, num_stars, image.shape))
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

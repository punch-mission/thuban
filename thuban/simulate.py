import numpy as np
from astropy.table import QTable
from photutils.datasets import make_gaussian_sources_image, make_noise_image

from thuban.catalog import filter_for_visible_stars, find_catalog_in_image


def simulate_star_image(catalog, wcs, img_shape, fwhm,
                        distortion_x_shift=None,
                        distortion_y_shift=None,
                        mag_set=0, flux_set=500_000, noise_mean=25.0, noise_std=5.0, dimmest_magnitude=8):
    sigma = fwhm / 2.355

    stars = find_catalog_in_image(filter_for_visible_stars(catalog, dimmest_magnitude=dimmest_magnitude),
                                  wcs, img_shape)
    star_mags = stars['Vmag']

    sources = QTable()
    sources['x_mean'] = stars['x_pix']
    sources['y_mean'] = stars['y_pix']
    sources['x_stddev'] = np.ones(len(stars))*sigma
    sources['y_stddev'] = np.ones(len(stars))*sigma
    sources['flux'] = flux_set * np.power(10, -0.4*(star_mags - mag_set))
    sources['theta'] = np.zeros(len(stars))

    if distortion_x_shift is not None and distortion_y_shift is not None:
        new_x = sources['x_mean'] + distortion_x_shift(sources['x_mean'], sources['y_mean'])
        new_y = sources['y_mean'] + distortion_y_shift(sources['x_mean'], sources['y_mean'])
        sources['x_mean'] = new_x
        sources['y_mean'] = new_y

    fake_image = make_gaussian_sources_image(img_shape, sources)
    fake_image += make_noise_image(img_shape, 'gaussian', mean=noise_mean, stddev=noise_std)

    return fake_image, sources

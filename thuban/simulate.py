import numpy as np
from astropy.table import QTable
from photutils.datasets import make_gaussian_sources_image, make_noise_image

from thuban.catalog import (filter_for_visible_stars, find_catalog_in_image,
                            load_hipparcos_catalog)


def simulate_star_image(wcs, img_shape, fwhm,
                        wcs_mode: str = 'all',
                        mag_set=0,
                        flux_set=500_000,
                        noise_mean: float | None = 25.0,
                        noise_std: float | None = 5.0,
                        dimmest_magnitude=8):
    sigma = fwhm / 2.355

    catalog = load_hipparcos_catalog()
    filtered_catalog = filter_for_visible_stars(catalog,
                                                dimmest_magnitude=dimmest_magnitude)
    stars = find_catalog_in_image(filtered_catalog,
                                  wcs,
                                  img_shape,
                                  mode=wcs_mode)
    star_mags = stars['Vmag']

    sources = QTable()
    sources['x_mean'] = stars['x_pix']
    sources['y_mean'] = stars['y_pix']
    sources['x_stddev'] = np.ones(len(stars))*sigma
    sources['y_stddev'] = np.ones(len(stars))*sigma
    sources['flux'] = flux_set * np.power(10, -0.4*(star_mags - mag_set))
    sources['theta'] = np.zeros(len(stars))

    fake_image = make_gaussian_sources_image(img_shape, sources)
    if noise_mean is not None and noise_std is not None:  # we only add noise if it's specified
        fake_image += make_noise_image(img_shape, 'gaussian', mean=noise_mean, stddev=noise_std)

    return fake_image, sources

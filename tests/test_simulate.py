import numpy as np
from astropy.wcs import WCS

from thuban.simulate import simulate_star_image


def test_simulate_star_image():
    shape = (512, 1024)
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    w.wcs.cdelt = np.array([-0.1, 0.1])
    w.wcs.crval = [180, 0]
    w.wcs.ctype = ["RA", "DEC"]
    w.wcs.cunit = "deg", "deg"

    image, sources = simulate_star_image(w, shape, 3.0, flux_set=1E-8, noise_mean=None, noise_std=None)
    assert image.shape == shape
    print(image.max())

    import matplotlib.pyplot as plt
    plt.imshow(image, vmin=0, vmax=1E-10)
    plt.savefig("test.png")

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
from photutils.detection import IRAFStarFinder

from thuban.catalog import load_hipparcos_catalog, match_catalog_to_sources
from thuban.simulate import simulate_star_image


def get_simulated_image():
    shape = (1024, 1024)
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    w.wcs.cdelt = np.array([0.05, 0.05])
    w.wcs.crval = [240, -60]
    w.wcs.ctype = ["RA", "DEC"]
    w.wcs.cunit = "deg", "deg"

    image, sources = simulate_star_image(w, shape, 3.0, flux_set=1E-8, noise_mean=None, noise_std=None)
    return image, w

def get_real_image():
    # path = "/Users/jhughes/Downloads/0/PP3/2025/02/05/PUNCH_L0_PP3_20250205073002_v1.fits"
    path = "/Users/jhughes/Downloads/0/PP4/2025/03/01/PUNCH_L0_PP4_20250301014806_v1.fits"
    key = "A"
    hdu_num = 1
    #path = "/Users/jhughes/Desktop/data/WFI_campaign2_night2_phase3_calibrated/campaign2_night2_phase3_calibrated_041.fits"
    #path = "/Users/jhughes/Desktop/data/WFI_campaign2_night2_phase3_calibrated/campaign2_night2_phase3_calibrated_218.fits"
    # key = " "
    # hdu_num = 0
    with fits.open(path) as hdu:
        data = hdu[hdu_num].data
        wcs = WCS(hdu[hdu_num].header, hdu, key=key)

    return data, wcs




if __name__ == "__main__":
    # data, w = get_simulated_image()
    # vmin, vmax = 0, 5E-11

    data, w = get_real_image()
    vmin, vmax = 0, 200

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    print(np.array((mean, median, std)))

    # daofind = DAOStarFinder(fwhm=3.5, threshold=5. * std)
    daofind = IRAFStarFinder(fwhm=3.5, threshold=1. * std)

    sources = daofind(data - median)
    for col in sources.colnames:
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'  # for consistent table output
    sources.pprint(max_width=180)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=4.0)
    plt.imshow(data, cmap='Greys_r', origin='lower', vmin=vmin, vmax=vmax)
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    plt.show()

    # create catalog of stars found in image that are correlated with hipparcos
    catalog = load_hipparcos_catalog()
    # reduced_catalog = find_catalog_in_image(catalog, w, data.shape)
    paired_data = match_catalog_to_sources(catalog, w, data.shape, sources)
    paired_data.pprint(max_width=76)
    # take pairs of stars and calculate their distance in pixels and sky
    estimated_platescale = []
    position = []
    for _ in range(5_000):
        first = np.random.randint(len(paired_data))
        second = np.random.randint(len(paired_data))
        if first != second:
            first = paired_data[first]
            second = paired_data[second]
            pixel_distance = np.sqrt(np.square(first["xcentroid"] - second["xcentroid"]) + np.square(first["ycentroid"] - second["ycentroid"]))

            # first_coord = SkyCoord(catalog[catalog['HIP'] == first['HIP']]['RAdeg'],
            #                        catalog[catalog['HIP'] == first['HIP']]['DEdeg'], unit='deg')
            # second_coord = SkyCoord(catalog[catalog['HIP'] == second['HIP']]['RAdeg'],
            #                         catalog[catalog['HIP'] == second['HIP']]['DEdeg'], unit='deg')
            # angular_distance = first_coord.separation(second_coord)

            first_coord = np.array([catalog[catalog['HIP'] == first['HIP']]['RAdeg'],
                                   catalog[catalog['HIP'] == first['HIP']]['DEdeg']])
            second_coord = np.array([catalog[catalog['HIP'] == second['HIP']]['RAdeg'],
                                    catalog[catalog['HIP'] == second['HIP']]['DEdeg']])
            angular_distance = np.sqrt(np.square(first_coord[0] - second_coord[0]) + np.square(first_coord[1] - second_coord[1]))

            estimated_platescale.append(angular_distance / pixel_distance)
            position.append(np.array([np.mean([first["xcentroid"], second["xcentroid"]]),
                                      np.mean([first["ycentroid"], second["ycentroid"]])]))
    position = np.array(position)
    estimated_platescale = np.array(estimated_platescale)
    print(np.mean(estimated_platescale), np.std(estimated_platescale))

    half_window = 20
    center = [c // 2 for c  in data.shape]
    x_mask = (position[:, 0] > center[0] - half_window) * (position[:, 0] < center[0] + half_window)
    y_mask = (position[:, 1] > center[1] - half_window) * (position[:, 1] < center[1] + half_window)
    mask = x_mask * y_mask
    print("center", np.mean(estimated_platescale[mask]), np.std(estimated_platescale[mask]))

    fig, ax = plt.subplots()
    im = ax.scatter(position[:, 0], position[:, 1], c=estimated_platescale)
    plt.colorbar(im)
    plt.show()
    # see how the platescale varies across the image

    grid_x, grid_y = np.meshgrid(np.linspace(0, data.shape[0], 100),
                                 np.linspace(0, data.shape[1], 100))
    grid_result = scipy.interpolate.griddata(position, estimated_platescale, (grid_x, grid_y), method='cubic')

    fig, ax = plt.subplots()
    im = ax.imshow(grid_result, vmin=0.01, vmax=0.03)
    plt.colorbar(im)
    plt.show()
    print("grid mean", np.nanmean(grid_result))

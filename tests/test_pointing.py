# from astropy.io import fits
# from astropy.wcs import WCS
# import matplotlib.pyplot as plt
#
# from thuban.pointing import refine_pointing
# from thuban.catalog import load_hipparcos_catalog, filter_for_visible_stars, find_catalog_in_image
#
#
# def test_pointing_solving():
#     path = "/Users/jhughes/Desktop/data/data_star_removal/hi1_bgsub_psf_repoint2/20220913_012831_s4h1A.fts"
#     path = '/Users/jhughes/Desktop/data/data_star_removal/hi1_bgsub_psf_repoint2/20220822_212831_s4h1A.fts'
#     # path = "/Users/jhughes/Desktop/data/data_star_removal/hi1_bgsub_psf_repoint2/20220913_072831_s4h1A.fts"
#     # path = "/Users/jhughes/Desktop/data/data_star_removal/hi1_bgsub_psf_repoint2/20220918_212831_s4h1A.fts"
##
#     with fits.open(path) as hdul:
#         w = WCS(hdul[0].header, hdul, key='A')
#         d = hdul[0].data
#         i = hdul[0].data.byteswap().newbyteorder().astype(float)
#
#     num_stars = 50
#     w2, observed, r, num_trials = refine_pointing(i, w,
#                                   detection_threshold=100,
#                                   method='least_squares',
#                                   dimmest_magnitude=7.0,
#                                   num_stars=num_stars,
#                                   chisqr_threshold=0.05,
#                                          max_trials=10
#                                   )
#
#     print(r)
#
#     catalog = load_hipparcos_catalog()
#     visible_stars = filter_for_visible_stars(catalog, dimmest_magnitude=8.0)
#     stars_in_image = find_catalog_in_image(visible_stars, w2, d.shape)
#
#     fig, ax = plt.subplots()
#     ax.imshow(d, vmin=0, vmax=10_000, cmap='Greys_r')
#     ax.plot(observed[:, 0], observed[:, 1], 'r.')
#     ax.plot(stars_in_image['x_pix'][:num_stars], stars_in_image['y_pix'][:num_stars], 'y*')
#     plt.show()
#
#     print("done")

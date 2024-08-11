import concurrent.futures
import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import sep_pjw as sep
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

from thuban.catalog import (filter_for_visible_stars, find_catalog_in_image,
                            load_hipparcos_catalog)
from thuban.distortion import compute_distortion, make_empty_distortion_model
from thuban.pointing import (convert_cd_matrix_to_pc_matrix,
                             refine_pointing_wrapper)
from thuban.util import find_celestial_wcs, remove_pairless_points


def _flatten_list(xss):
    return [x for xs in xss for x in xs]


def get_files(directory: Path, valid_extensions=("fits", "fts")) -> [Path]:
    valid_extensions = ["*." + ext for ext in valid_extensions]
    if directory.is_dir():
        return sorted(_flatten_list([list(directory.glob(ext)) for ext in valid_extensions]))
    else:  # noqa RET505
        raise RuntimeError(f"Specified {directory} is not a directory")


def open_files(paths: [Path], byte_swap=True) -> (np.ndarray, [fits.Header]):
    with fits.open(paths[0]) as hdul:
        data_hdu_num, celestial_wcs_key = find_celestial_wcs(hdul)

    all_data, all_headers, all_wcses = [], [], []
    for filename in tqdm(paths):
        with fits.open(filename) as hdul:
            if byte_swap:
                data = hdul[data_hdu_num].data.byteswap().newbyteorder().astype(float)
            else:
                data = hdul[data_hdu_num].data.astype(float)
            head = hdul[data_hdu_num].header
            all_wcses.append(WCS(head, hdul, key=celestial_wcs_key))
        all_data.append(data)
        all_headers.append(head)
    return np.array(all_data), all_headers, celestial_wcs_key, all_wcses


@click.command()
@click.argument("directory", type=click.Path())
@click.option("-b", "--byte_swap", default=True, is_flag=True)
@click.option("-m", "--magnitude", default=7.0, type=float)
@click.option("--num_bins", default=75, type=int)
@click.option("-i", "--iterations", default=5, type=int)
@click.option("-w", "--workers", default=16, type=int)
@click.option("-n", "--num-stars", default=20, type=int)
@click.option("--background", default=16, type=int)
@click.option("--threshold", default=5, type=float)
@click.option("--skip-distortion", default=False, is_flag=True)
def determine_pointing_and_distortion(directory, byte_swap=True,  num_stars=20,
                                      magnitude=7.0, num_bins=75, iterations=5, workers=16,
                                      background=16, threshold=5.0, skip_distortion=False):
    start = datetime.now()

    filenames = get_files(Path(directory))
    print(f"{len(filenames)} FITS files found for processing")

    if len(filenames) == 0:
        print("No files found. Quitting.")
        return

    print("Opening files")
    data, headers, celestial_wcs_key, all_wcses = open_files(filenames, byte_swap)

    # edge = 100
    # data[:, :, :edge] = 0.0
    # data[:, :edge, :] = 0.0
    # data[:, -edge:, :] = 0.0
    # data[:, :, -edge:] = 0.0

    print(headers)
    current_wcses = [convert_cd_matrix_to_pc_matrix(w) for w in all_wcses]

    # remove any SIP distortion model and make an empty table
    cpdis1, cpdis2 = make_empty_distortion_model(num_bins, data[0])
    for wcs in current_wcses:
        if '-SIP' in [name[-4:] for name in wcs.wcs.ctype]:
            wcs.wcs.ctype = [name[:-4] for name in wcs.wcs.ctype]
        wcs.sip = None
        wcs.cpdis1 = cpdis1
        wcs.cpdis2 = cpdis2
    print(current_wcses[0].wcs.cunit)

    counts = []
    for iteration in range(iterations):
        print(f"ITERATION {iteration} \n\tpreparing")
        all_found_positions = []
        all_no_distortion = []

        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

        futures = []
        for file_index in tqdm(range(len(filenames))):
            futures.append(
                executor.submit(refine_pointing_wrapper,
                                data[file_index],
                                current_wcses[file_index],
                                file_index,
                                detection_threshold=threshold,
                                max_trials=5,
                                dimmest_magnitude=magnitude,
                                num_stars=num_stars,
                                background_width=background,
                                background_height=background,
                                chisqr_threshold=0.05,
                                method='least_squares'))

        print("\tprocessing")
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):

                updated_wcs, _, solution, num_trials, file_index = future.result()
                current_wcses[file_index] = updated_wcs


                bg = sep.Background(data[file_index], bw=background, bh=background)
                data_sub = data[file_index] - bg
                objects = sep.extract(data_sub, threshold, err=bg.globalrms)
                observed_coords = np.stack([objects["x"], objects["y"]], axis=-1)

                new_stars = find_catalog_in_image(
                    filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=8.0),
                    updated_wcs, data.shape[1:], mode='wcs')
                all_no_distortion.append(np.stack([new_stars['x_pix'], new_stars['y_pix']], axis=-1))

                all_found_positions.append(observed_coords)
                # all_no_distortion.append(no_distortion)
                pbar.update(1)

        # log.append((all_corrected_positions, all_found_positions))
        # TODO: move lower to fix counting of stars
        count = len(all_no_distortion)
        counts.append(count)

        final_observations, final_no_distortion = [], []
        for i in range(len(all_no_distortion)):
            observed_filtered, refined_filtered = remove_pairless_points(all_found_positions[i],
                                                                         all_no_distortion[i],
                                                                         max_distance=15)
            if (observed_filtered.ndim == 2 and refined_filtered.ndim == 2
                    and len(observed_filtered) > 0 and len(refined_filtered) > 0):
                final_observations.append(observed_filtered)
                final_no_distortion.append(refined_filtered)
        final_observations = np.concatenate(final_observations, axis=0)
        final_refined = np.concatenate(final_no_distortion, axis=0)

        print("star count:", final_refined.shape[0])

        if not skip_distortion:
            cpdis1, cpdis2 = compute_distortion(data[0].shape,
                                                        final_observations,
                                                        final_refined,
                                                        num_bins=num_bins)

            for wcs in current_wcses:
                wcs.cpdis1 = cpdis1
                wcs.cpdis2 = cpdis2

    end = datetime.now()
    print(f"Completed in {end - start} with {sum(counts)} stars")

    print("Writing results to file")
    for fn, updated_wcs, img in zip(filenames, current_wcses, data):
        _, extension = os.path.splitext(fn)
        hdul = updated_wcs.to_fits()  # improve the writing mechanism
        hdul[0].data = img
        hdul.writeto(str(fn).replace(extension, "_repoint.fits"), overwrite=True)  # todo make postfix and option

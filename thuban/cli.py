import concurrent.futures
import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import sep
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

from thuban.catalog import filter_for_visible_stars, load_hipparcos_catalog
from thuban.distortion import compute_distortion, make_empty_distortion_model
from thuban.pointing import (convert_cd_matrix_to_pc_matrix, get_star_lists,
                             refine_pointing)
from thuban.util import find_celestial_wcs


def _flatten_list(xss):
    return [x for xs in xss for x in xs]


def get_files(directory: Path, valid_extensions=("fits","fts")) -> [Path]:
    valid_extensions = ["*." + ext for ext in valid_extensions]
    if directory.is_dir():
        return sorted(_flatten_list([list(directory.glob(ext)) for ext in valid_extensions]))
    else:  # noqa RET505
        raise RuntimeError(f"Specified {directory} is not a directory")


def open_files(paths: [Path], byte_swap=True) -> (np.ndarray, [fits.Header]):
    with fits.open(paths[0]) as hdul:
        data_hdu_num, celestial_wcs_key = find_celestial_wcs(hdul)

    all_data, all_headers = [], []
    for filename in tqdm(paths):
        with fits.open(filename) as hdul:
            if byte_swap:
                data = hdul[data_hdu_num].data.byteswap().newbyteorder().astype(float)
            else:
                data = hdul[data_hdu_num].data.astype(float)
            head = hdul[data_hdu_num].header
        all_data.append(data)
        all_headers.append(head)
    return np.array(all_data), all_headers, celestial_wcs_key


@click.command()
@click.argument("directory", type=click.Path())
@click.option("-b", "--byte_swap", default=True, is_flag=True)
@click.option("-m", "--magnitude", default=8.0, type=float)
@click.option("-l", "--limit", default=1.0, type=float)
@click.option("--num_bins", default=75, type=int)
@click.option("-i", "--iterations", default=5, type=int)
@click.option("-w", "--workers", default=16, type=int)
@click.option("--background", default=16, type=int)
@click.option("--threshold", default=5, type=float)
@click.option("--skip-distortion", default=False, is_flag=True)
def determine_pointing_and_distortion(directory, byte_swap=True,
                                      magnitude=8.0, limit=1.0, num_bins=75, iterations=5, workers=16,
                                      background=16, threshold=5.0, skip_distortion=False):
    start = datetime.now()

    filenames = get_files(Path(directory))
    print(f"{len(filenames)} FITS files found for processing")

    if len(filenames) == 0:
        print("No files found. Quitting.")
        return

    print("Opening files")
    data, headers, celestial_wcs_key = open_files(filenames, byte_swap)

    print("Filtering catalog")
    catalog = load_hipparcos_catalog()
    visible_stars = filter_for_visible_stars(catalog, dimmest_magnitude=magnitude)

    print(headers)
    current_wcses = [convert_cd_matrix_to_pc_matrix(WCS(head, key=celestial_wcs_key)) for head in headers]

    # remove any SIP distortion model and make an empty table
    cpdis1, cpdis2 = make_empty_distortion_model(num_bins, data[0])
    for wcs in current_wcses:
        if '-SIP' in [name[-4:] for name in wcs.wcs.ctype]:
            wcs.wcs.ctype = [name[:-4] for name in wcs.wcs.ctype]
        wcs.sip = None
        wcs.cpdis1 = cpdis1
        wcs.cpdis2 = cpdis2
    print(current_wcses[0].wcs.cunit)

    log = []
    counts = []
    for iteration in range(iterations):
        print(f"ITERATION {iteration} \n\tpreparing")
        all_corrected_positions, all_found_positions = [], []
        all_no_distortion = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = []
        for file_index in tqdm(range(len(filenames))):
            bkg = sep.Background(data[file_index], bw=background, bh=background)
            d = data[file_index] - bkg
            observed_table = sep.extract(d, threshold, err=bkg.globalrms)
            observed_coords = np.stack([observed_table['x'], observed_table['y']], axis=-1)

            futures.append(
                executor.submit(refine_pointing,
                                data[file_index],
                                current_wcses[file_index],
                                file_index,
                                observed_coords,
                                visible_stars))

        print("\tprocessing")
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                updated_wcs, file_index, observed_coords = future.result()
                current_wcses[file_index] = updated_wcs
                found, refined_coords, no_distortion = get_star_lists(visible_stars,
                                                                      current_wcses[file_index],
                                                                      observed_coords)

                all_corrected_positions.append(refined_coords)
                all_found_positions.append(found)
                all_no_distortion.append(no_distortion)
                pbar.update(1)

        log.append((all_corrected_positions, all_found_positions))
        all_corrected_positions = np.concatenate(all_corrected_positions)
        all_found_positions = np.concatenate(all_found_positions)
        all_no_distortion = np.concatenate(all_no_distortion)
        count = len(all_corrected_positions)
        counts.append(count)

        if not skip_distortion:
            cpdis1, cpdis2 = compute_distortion(data[0].shape,
                                                        all_found_positions,
                                                        all_no_distortion,
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
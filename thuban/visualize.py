import matplotlib.pyplot as plt
import numpy as np


def plot_result_pointing_and_distortion(
    image,
    old_positions,
    new_positions_no_distort,
    new_positions,
    wcs,
    limit=3.5,
    vmin=0,
    vmax=20_000**2,
    cmap="Greys_r",
    figsize=(15, 15),
):
    dx, dy = wcs.cpdis1.data, wcs.cpdis2.data
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=figsize, sharex=True, sharey=True)
    im_star = axs[0, 0].imshow(image**2, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap, interpolation="None")
    axs[0, 0].plot(old_positions[:, 0], old_positions[:, 1], "r*", label="original")
    axs[0, 0].plot(new_positions_no_distort[:, 0], new_positions_no_distort[:, 1], "y*", label="updated no distortion")
    axs[0, 0].plot(new_positions[:, 0], new_positions[:, 1], "g*", label="fully updated")
    axs[0, 0].legend(loc=1)
    axs[0, 0].set_xlim((0, image.shape[0]))
    axs[0, 0].set_ylim((0, image.shape[1]))
    axs[0, 0].set_title("Starfield with positions")
    fig.colorbar(im_star)

    im0 = axs[1, 0].imshow(
        dx, vmin=-limit, vmax=limit, origin="lower", cmap="seismic", extent=[0, image.shape[0], 0, image.shape[1]]
    )
    axs[1, 0].set_title("x distortion")
    fig.colorbar(im0)

    im1 = axs[1, 1].imshow(
        dy, vmin=-limit, vmax=limit, origin="lower", cmap="seismic", extent=[0, image.shape[0], 0, image.shape[1]]
    )
    axs[1, 1].set_title("y distortion")
    fig.colorbar(im1)

    im2 = axs[0, 1].imshow(
        np.sqrt(np.square(dx) + np.square(dy)),
        vmin=0,
        vmax=limit,
        origin="lower",
        extent=[0, image.shape[0], 0, image.shape[1]],
    )
    axs[0, 1].set_title("distortion magnitude")
    fig.colorbar(im2)

    return fig, axs

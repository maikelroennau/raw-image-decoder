import argparse
import ast
import logging
from pathlib import Path

import imageio
import numpy as np
import rawpy
from numpy import matlib
from scipy.ndimage.filters import convolve


def demosaic(image, extension, output="results"):
    logging.info("START demosicing")
    red_filter = np.array(
        [[1, 0],
         [0, 0]])

    green_filter = np.array(
        [[0, 1],
         [1, 0]])

    blue_filter = np.array(
        [[0, 0],
         [0, 1]])

    # Replicate pattern over to the image shape
    height, width = image.shape
    red_mask = matlib.repmat(red_filter, height//2, width//2)
    geen_mask = matlib.repmat(green_filter, height//2, width//2)
    blue_mask = matlib.repmat(blue_filter, height//2, width//2)

    # Save individual color channels for visualization
    color_names = ["red", "green", "blue"]
    color_filters = [red_mask, geen_mask, blue_mask]

    for i, (name, filter) in enumerate(zip(color_names, color_filters)):
        channel_image = np.zeros((height, width, 3))
        channel_image[:, :, i] = image * filter
        imageio.imwrite(
            str(Path(output).joinpath(f"0{i+2}_{name}{extension}")),
            (channel_image * 255).astype(np.uint8)
        )

    # Filter references: https://www.dmi.unict.it/~battiato/EI_MOBILE0708/Color%20interpolation%20(Guarnera).pdf, page 12-13.
    red_blue_interpolation = np.array(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]) / 4

    green_interpolation = np.array(
        [[0, 1, 0],
         [1, 4, 1],
         [0, 1, 0]]) / 4

    # Perform demosicing buy interpolating pixel color with a convolution
    demosaiced = np.zeros((height, width, 3))
    demosaiced[:, :, 0] = convolve(image * red_mask, red_blue_interpolation)
    demosaiced[:, :, 1] = convolve(image * geen_mask, green_interpolation)
    demosaiced[:, :, 2] = convolve(image * blue_mask, red_blue_interpolation)

    logging.info("END demosicing")
    return demosaiced


def white_balance(image, white_reference):
    logging.info("START white_balance")
    if isinstance(white_reference, list):
        # If an array, assume values are already standardized.
        reference = np.array(white_reference)
    elif isinstance(white_reference, tuple):
        # If a pixel reference, standardize the values.
        reference = np.array([
            1./image[white_reference[0], white_reference[1], 0],
            1./image[white_reference[0], white_reference[1], 1],
            1./image[white_reference[0], white_reference[1], 2]
        ])

    image = np.clip(image * reference, 0., 1.)
    logging.info("END white_balance")
    return image


def gamma_correction(image, gamma):
    logging.info("START gamma_correction")
    image = np.power(image, gamma)
    logging.info("END gamma_correction")
    return image


def decode(image_path, white_reference=(1200, 230), gamma=1./2.2, extension=".png", output="results"):
    raw = rawpy.imread(image_path)
    rescaled = (raw.raw_image_visible - raw.raw_image_visible.min()) / (raw.raw_image_visible.max() - raw.raw_image_visible.min())

    output_path = Path(output)
    output_path.mkdir(exist_ok="True")
    imageio.imwrite(str(output_path.joinpath(f"01_scene_raw{extension}")), (rescaled * 255).astype(np.uint8))

    demosaiced = demosaic(rescaled, extension, output)
    imageio.imwrite(str(output_path.joinpath(f"05_demoseiced{extension}")), (demosaiced * 255).astype(np.uint8))

    while_balanced = white_balance(demosaiced.copy(), white_reference)
    imageio.imwrite(str(output_path.joinpath(f"06_wb{extension}")), (while_balanced * 255).astype(np.uint8))

    gamma_corrected = gamma_correction(while_balanced, gamma)
    imageio.imwrite(str(output_path.joinpath(f"07_gamma{extension}")), (gamma_corrected * 255).astype(np.uint8))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description="Postprocess raw `.dng` images.")

    parser.add_argument(
        "-i",
        "--image",
        help="Path to the `.dng` images to be processed.",
        required=True,
        type=str)

    parser.add_argument(
        "-w",
        "--white-reference",
        help="""Tuple with X and Y coordinates or list of RGB values to be used as the white color reference.
                Examples: `(1200,230)`, `[1.25,1.0,0.8]`.""",
        default="1200,230",
        type=str)

    parser.add_argument(
        "-g",
        "--gamma",
        help="""Gamma value to apply for gamma correction. Should be provided in fraction format.
                Example: `numerator/denominator`. Default `1./2.2`.""",
        default="1./2.2",
        type=str)

    parser.add_argument(
        "-o",
        "--output",
        help="Path where to save the results to.",
        default="results",
        type=str)

    parser.add_argument(
        "-e",
        "--extension",
        help="The output extension of the postprocessed results. Default `.jpg`.",
        default=".png",
        type=str)

    args = parser.parse_args()

    gamma = args.gamma.split("/")
    args.gamma = float(gamma[0]) / float(gamma[1])

    decode(
        args.image,
        ast.literal_eval(args.white_reference),
        args.gamma,
        args.extension,
        args.output)

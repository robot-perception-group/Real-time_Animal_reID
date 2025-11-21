"""
image_processing.py

Functions for general image processing, including loading, resizing, converting to
grayscale, and normalizing images.
"""

import cv2
import numpy as np


def load_img(img_path: str, show: bool = False) -> np.ndarray:
    """
    Loads an image from the specified file path.

    Args:
    - img_path (str): Path to the image file.
    - show (bool): If True, the image will be displayed in a window.

    Returns:
    - img (numpy.ndarray): Loaded image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at {img_path}")
    if show:
        cv2.imshow("Loaded Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


def grayscale(cv2_img: np.ndarray, show: bool = False) -> np.ndarray:
    """
    Converts a color image to grayscale using efficient in-place conversion.

    Args:
    - cv2_img (np.ndarray): Input image in BGR format.
    - show (bool): If True, the grayscale image will be displayed.

    Returns:
    - np.ndarray: Grayscale image.
    """
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    if show:
        cv2.imshow("Grayscale Image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return gray


def resize_fix_ar(
    cv2_img: np.ndarray, desired_width: int, show: bool = False
) -> np.ndarray:
    """
    Resizes an image while maintaining the aspect ratio, with dynamic interpolation
    method selection based on scaling direction.

    Args:
    - cv2_img (np.ndarray): Input image to be resized.
    - desired_width (int): Desired width of the output image.
    - show (bool): If True, the resized image will be displayed.

    Returns:
    - np.ndarray: Resized image.
    """
    (h, w) = cv2_img.shape[:2]
    aspect_ratio = h / w
    new_height = int(desired_width * aspect_ratio)

    # Check if downscaling or upscaling
    if desired_width < w:
        # Downscaling: Use INTER_AREA (best for shrinking)
        interpolation_method = cv2.INTER_AREA
    else:
        # Upscaling: Use INTER_CUBIC (best for enlarging)
        interpolation_method = cv2.INTER_CUBIC

    resized = cv2.resize(
        cv2_img, (desired_width, new_height), interpolation=interpolation_method
    )

    if show:
        cv2.imshow("Resized Image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return resized


def normalize(cv2_img: np.ndarray, show: bool = False) -> np.ndarray:
    """
    Normalizes an image in-place to the given range [0, 255].

    Args:
    - cv2_img (np.ndarray): Input image to be normalized.
    - show (bool): If True, the normalized image will be displayed.

    Returns:
    - np.ndarray: Normalized image.
    """
    cv2.normalize(cv2_img, cv2_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    if show:
        cv2.imshow("Normalized Image", cv2_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cv2_img


def preproc(bobox_path: str) -> np.ndarray:
    """
    Processes an image by loading, resizing, converting to grayscale, and normalizing
    in the most efficient way possible.

    Args:
    - bobox_path (str): Path to the image file to be processed.

    Returns:
    - np.ndarray: Processed image.
    """
    raw_bobox = load_img(bobox_path, show=False)
    bobox = resize_fix_ar(raw_bobox, 224, show=False)
    bobox = grayscale(bobox, show=False)
    bobox = normalize(bobox, show=False)
    return bobox

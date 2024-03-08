import cv2
import numpy as np


def load_image(image_path=None):
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array, or None if an error occurs.

    Raises:
        ValueError: If no image path is provided.
        FileNotFoundError: If the image file is not found at the specified path.
        cv2.error: If there is an error loading the image using OpenCV.
    """
    if image_path is None:
        raise ValueError("No image path provided.")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    return image


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Applies Gaussian blur to an image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        kernel_size (tuple): The size of the Gaussian kernel. Default is (5, 5).
            The tuple should contain odd numbers.

    Returns:
        numpy.ndarray: The blurred image as a NumPy array.

    Raises:
        ValueError: If the input image is empty or None, or if the kernel size values are not odd numbers.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be empty or None.")

    kx, ky = kernel_size
    if kx % 2 == 0 or ky % 2 == 0:
        raise ValueError("Kernel size values must be odd numbers.")

    return cv2.GaussianBlur(image, kernel_size, 0)


def binarize_image(image):
    """
    Binarizes an input grayscale image using Otsu's thresholding method.

    Args:
        image (numpy.ndarray): The input grayscale image as a NumPy array.

    Returns:
        numpy.ndarray: The binarized image as a NumPy array.

    Raises:
        ValueError: If the input image is not grayscale.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def resize_image(image, target_height=512, inter=cv2.INTER_AREA):
    """
    Resizes an input image to a fixed height while maintaining its aspect ratio.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        target_height (int): The target height of the resized image. Default is 32.
        inter (int): The interpolation method used for resizing. Default is cv2.INTER_AREA.

    Returns:
        numpy.ndarray: The resized image as a NumPy array.

    Raises:
        ValueError: If the input image is empty or None.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be empty or None.")

    (h, w) = image.shape[:2]

    r = target_height / float(h)
    target_width = int(w * r)

    resized = cv2.resize(image, (target_width, target_height), interpolation=inter)

    return resized


def apply_morphological_operations(image, kernel_size=(1, 1), iterations=1):
    """
    Applies morphological opening and closing operations to a binary image.

    Args:
        image (numpy.ndarray): The input binary image as a NumPy array.
        kernel_size (tuple): The size of the kernel for the morphological operations. Default is (3, 3).
        iterations (int): The number of iterations for the morphological operations. Default is 1.

    Returns:
        numpy.ndarray: The image after applying morphological operations.

    Raises:
        ValueError: If the input image is not binary.
    """
    if len(image.shape) != 2 or image.dtype != np.uint8 or np.unique(image).size != 2:
        raise ValueError("Input image must be a binary image.")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return image


def noise_reduction(image, max_kernel_size=5):
    """
    Applies adaptive noise reduction to a grayscale image using median blur.
    
    The noise level is estimated using the standard deviation of the image intensities,
    and the kernel size for the median blur is determined based on the estimated noise level.

    Args:
        image (numpy.ndarray): The input grayscale image as a NumPy array.
        max_kernel_size (int): The maximum kernel size for the median blur. Default is 5.

    Returns:
        numpy.ndarray: The image after applying noise reduction.

    Raises:
        ValueError: If the input image is not grayscale.
    """
    if len(image.shape) != 2 or image.dtype != np.uint8:
        raise ValueError("Input image must be a grayscale image.")

    # Estimate the noise level using the standard deviation of image intensities
    noise_level = np.std(image) / 255.0

    # Calculate the kernel size based on the noise level and maximum kernel size
    kernel_size = int(noise_level * max_kernel_size)
    kernel_size = max(3, kernel_size)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Apply median blur with the calculated kernel size
    denoised_image = cv2.medianBlur(image, kernel_size)

    return denoised_image


def skew_correction(image):
    """
    Applies skew correction to a grayscale image using Hough line transform.

    Args:
        image (numpy.ndarray): The input grayscale image as a NumPy array.

    Returns:
        numpy.ndarray: The skew-corrected image.
    """
    # Create a binary mask of the image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect lines using Hough line transform
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Calculate the average skew angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    if len(angles) > 0:
        skew_angle = np.mean(angles)
    else:
        skew_angle = 0

    # Create an affine transformation matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

    # Apply the affine transformation to correct the skew
    corrected_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return corrected_image


def normalize_image(image):
    """
    Normalizes the pixel values of an image to the range [0, 1].

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The normalized image.
    """
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized_image


def add_padding(image, padding=20):
    """
    Adds padding around the image.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        padding (int): The amount of padding to add on each side of the image. Default is 20.

    Returns:
        numpy.ndarray: The padded image.
    """
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    return padded_image


def preprocess_image(image):  
    # Apply skew correction
    skew_corrected_image = skew_correction(image)
    
    # Apply noise reduction
    denoised_image = noise_reduction(skew_corrected_image)
    
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(denoised_image)
    
    # Binarize the image
    binarized_image = binarize_image(blurred_image)
    
    # Apply morphological operations
    morphed_image = apply_morphological_operations(binarized_image)
    
    # Resize the image
    resized_image = resize_image(morphed_image)
    
    # Normalize the pixel values
    normalized_image = normalize_image(resized_image)
    
    # Add padding if required
    padded_image = add_padding(normalized_image)
    
    return padded_image
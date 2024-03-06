import cv2
import numpy as np

def load_image(image_path):
    """Loads an image, handling potential errors"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def convert_to_grayscale(image):
    """Converts an image to grayscale"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian blur to an image"""
    return cv2.GaussianBlur(image, kernel_size, 0)

def binarize_image(image):
    """Binarize an image using Otsu's thresholding"""
    _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize an image while maintaining its aspect ratio"""
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def apply_morphological_operations(image, kernel_size=(3, 3), iterations=1):
    """Applies morphological operations to an image"""
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return image

def noise_reduction(image, max_kernel_size=5):
    """Reduces noise in the image adaptively"""
    noise_level = estimate_noise_level(image)
    kernel_size = int(noise_level * max_kernel_size)
    kernel_size = max(3, kernel_size) 
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1 # Ensure odd kernel size
    return cv2.medianBlur(image, kernel_size)

def estimate_noise_level(image):
    """Estimates the noise level in the image"""
    return np.std(image) / 255.0

def preprocess_image(image_path, output_width=None, output_height=None):
    """Preprocess an image by applying several preprocessing steps."""
    image = load_image(image_path)
    if image is None:
        return None

    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    morph = apply_morphological_operations(blurred)
    noise_reduced = noise_reduction(morph)
    binarized = binarize_image(noise_reduced)[1]

    if output_width or output_height:
        resized = resize_image(binarized, width=output_width, height=output_height)
        return resized

    return binarized

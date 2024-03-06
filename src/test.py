import cv2
from preprocess import *

image = load_image("../images/img.png")

if image is not None:
    # Convert the image to grayscale
    gray_image = convert_to_grayscale(image)

    # Apply Gaussian blur to the image
    blurred_image = apply_gaussian_blur(gray_image, kernel_size=(7, 7))

    # Binarize the blurred image with a threshold of 127
    binarized_image = binarize_image(blurred_image)

    # Morphological operations
    morph_image = apply_morphological_operations(binarized_image, kernel_size=(3, 3), iterations=1)

    # Noise reduction
    noise_reduced_image = noise_reduction(morph_image, max_kernel_size=5)

    # Resize the image
    resized_image = resize_image(noise_reduced_image, width=500)

    # Display the pipeline results
    cv2.imshow("Original Image", image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow("Binarized Image", binarized_image)
    cv2.imshow("Morphological Operations", morph_image)
    cv2.imshow("Noise Reduced Image", noise_reduced_image)
    # cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image. Please check the file path.")
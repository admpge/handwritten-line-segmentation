import cv2
from preprocess import *

image = load_image("../images/sample_2_line.png")

if image is not None:
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
    
    # Add padding
    padded_image = add_padding(normalized_image)
    
    # Display the pipeline results
    cv2.imshow("Original Image", image)
    cv2.imshow("Skew Corrected Image", skew_corrected_image)
    cv2.imshow("Denoised Image", denoised_image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow("Binarized Image", binarized_image)
    cv2.imshow("Morphological Operations", morphed_image)
    cv2.imshow("Resized Image", resized_image)
    cv2.imshow("Normalized Image", normalized_image)
    cv2.imshow("Padded Image", padded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image. Please check the file path.")
# Handwritten Line Segmentation

This project aims to develop a system for segmenting handwritten text into individual lines using computer vision techniques. The line segmentation process is a crucial step in handwritten text recognition pipelines, as it enables further processing and analysis of the text at the line level.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Line Segmentation](#line-segmentation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Handwritten line segmentation is the process of dividing an image containing handwritten text into separate lines. This project implements various image processing techniques to accurately detect and extract individual lines from the input image. The segmented lines can then be used as input for subsequent steps in a handwritten text recognition pipeline, such as character segmentation and recognition.

## Installation
To use this project, you need to have Python installed on your system. You can clone the repository and install the required dependencies using the following commands:

git clone https://github.com/your-username/handwritten-line-segmentation.git
cd handwritten-line-segmentation
pip install -r requirements.txt

## Usage
To segment handwritten lines from an image, you can run the `line_segmentation.py` script with the following command:

python line_segmentation.py --input path/to/input/image.jpg --output path/to/output/directory

The script takes the following arguments:
- `--input`: Path to the input image file containing the handwritten text.
- `--output`: Path to the directory where the segmented line images will be saved.

## Preprocessing
Before performing line segmentation, the input image undergoes several preprocessing steps to enhance its quality and facilitate the segmentation process. The preprocessing steps include:
1. Grayscale conversion: The image is converted to grayscale to simplify further processing.
2. Resizing: The image is resized while maintaining its aspect ratio to ensure consistent input dimensions.
3. Noise reduction: Median blurring is applied to reduce noise and remove small artifacts from the image.
4. Gaussian blurring: A Gaussian blur is applied to further smooth the image and reduce high-frequency noise.
5. Binarization: The image is binarized using Otsu's thresholding method to separate the text from the background.
6. Morphological operations: Morphological operations, such as opening and closing, are applied to remove small artifacts and fill gaps in the text.

## Line Segmentation
The line segmentation process is based on the horizontal projection profile method. The steps involved in line segmentation are as follows:
1. Compute the horizontal projection profile of the preprocessed image.
2. Identify the valleys (local minima) in the projection profile, which represent the spaces between text lines.
3. Use the valleys as delimiters to extract individual line images from the original image.
4. Postprocess the extracted line images to remove any remaining noise or artifacts.

## Results
The segmented line images are saved in the specified output directory. Each line image is saved as a separate file with a unique identifier. These line images can be further processed for tasks such as character segmentation, feature extraction, and recognition.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License
This project is licensed under the [MIT License](LICENSE).
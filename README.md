# Image Difference and Background Removal

This repository contains two Python programs aimed at processing images:

1. **Image Difference and Bounding Box Highlighting**: This program, named `add_bounding_box.py`, detects differences between two input images and highlights them using bounding boxes.
2. **Background Removal**: The second program, named `delete_background.py`, removes the background from an image based on a specified bounding box.

## Example
Image modyfied with Kewin
<p align="center">
<img src="https://github.com/knrdsmt/Find-differences/blob/main/dublin_edited.jpg?raw=true" alt="Image modyfied with Kewin" width="95%" />
</p>
Bounding boxes marking changes
<p align="center">
<img src="https://github.com/knrdsmt/Find-differences/blob/main/dublin_bbox.jpg?raw=true" alt="Bounding boxes marking changes" width="95%" />
</p>
<p>&nbsp;</p>
Bounding box without background
<p align="left">
  <p>&nbsp;</p>
<img src="https://github.com/knrdsmt/Find-differences/blob/main/bbox_without_background.png?raw=true" alt="Bounding box without background" width="75" />
</p>
<p>&nbsp;</p>

---
## Image Difference and Bounding Box Highlighting

1. **Loading Images**: The program loads two input images using the OpenCV library (`cv2.imread`). These images are typically different versions of the same scene or object.
   
2. **Calculating Image Difference**: The absolute difference between the pixel values of the two images is calculated. This is done by converting the images to integers, taking their absolute difference, and then converting back to unsigned integers (`np.abs(image1.astype(int) - image2.astype(int)).astype(np.uint8)`).

3. **Grayscale Conversion**: The resulting difference image is converted to grayscale. This is achieved by taking the dot product of the difference image with a set of coefficients to obtain luminance values (`np.dot(difference[..., :3], [0.2989, 0.5870, 0.1140])`).

4. **Thresholding**: A threshold is applied to the grayscale image to identify significant differences between the two images. This creates a binary mask where white pixels represent areas of significant difference (`threshold = (gray_difference > 30) * 255`).

5. **Contour Detection**: Contours are detected within the thresholded image. Contours represent the boundaries of regions with continuous color or intensity. These contours will outline the areas of difference between the two images.

6. **Bounding Box Creation**: Bounding boxes are drawn around the detected contours. This is achieved by finding the minimum and maximum coordinates of each contour and drawing rectangles around them on the original image (`cv2.rectangle`).

7. **Output**: The modified image, with bounding boxes highlighting the differences between the two input images, is saved as `dublin_bbox.jpg`.

### Background Removal

1. **Reading Bounding Box Information**: The program reads the bounding box information generated by `add_bounding_box.py`. This information defines the region of interest (ROI) where differences were detected.

2. **Extracting Region of Interest**: Using the bounding box coordinates, the program extracts the region of interest (ROI) from the original images. This isolates the area where differences were detected.

3. **Difference Calculation**: The program calculates the difference between the original and modified images within the ROI. This identifies areas where the background was removed or altered.

4. **Mask Creation**: A binary mask is created based on the identified differences within the ROI. This mask distinguishes between foreground (areas of difference) and background.

5. **Background Removal**: The background outside the bounding box is removed from the ROI using the binary mask. This results in an image where only the foreground, representing the object or scene of interest, remains.

6. **Output**: The resulting image, with the background removed, is saved as `bbox_without_background.png`.

These programs utilize basic image processing techniques such as thresholding, contour detection, and masking to highlight differences between images and remove backgrounds. They can be further customized and optimized for specific applications and use cases.

## Instructions

### Image Difference and Bounding Box Highlighting

To use the `add_bounding_box.py` program:

1. Ensure you have Python installed on your system.
2. Install the required libraries using pip:

```bash
pip install opencv-python numpy
```

3. Place your input images in the same directory as the script.
4. Run the script and provide the filenames of the two images to compare:

```bash
python add_bounding_box.py dublin.jpg dublin_edited.jpg
```

5. The program will generate an output image named `dublin_bbox.jpg`, where the differences between the two images are highlighted with bounding boxes.

### Background Removal

To utilize the `delete_background.py` program:

1. Make sure Python is installed on your machine.
2. Install the required libraries using pip:

```bash
pip install opencv-python numpy
```

3. Ensure that the `add_bounding_box.py` program has been executed first to generate the bounding box information.
4. Run the `delete_background.py` script:

```bash
python delete_backgroung.py
```

5. The program will process the bounding box information generated by `add_bounding_box.py` and remove the background from the specified region.
6. The resulting image, `bbox_without_background.png`, will be saved in the same directory.

## Notes

- Both programs require the NumPy and OpenCV used only to read and write images. Ensure they are installed before running the scripts.
- Make sure the input images are in the same directory as the scripts or provide the correct paths when running the programs.
- Adjust parameters in the scripts if necessary, such as the threshold values for detecting differences or background removal.
- These programs are designed for basic image processing tasks and may require further customization for specific use cases.

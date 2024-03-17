import cv2
import numpy as np

image1 = cv2.imread('dublin.jpg')
image2 = cv2.imread('dublin_edited.jpg')

difference = np.abs(image1.astype(int) - image2.astype(int)).astype(np.uint8)
gray_difference = np.dot(difference[..., :3], [0.2989, 0.5870, 0.1140])
threshold = (gray_difference > 30) * 255

contours = []
for y in range(threshold.shape[0]):
    for x in range(threshold.shape[1]):
        if threshold[y, x] == 255:
            contours.append((x, y))

objects = []
for contour in contours:
    added = False
    for obj in objects:
        if min([np.sqrt((x - contour[0]) ** 2 + (y - contour[1]) ** 2) for x, y in obj]) < 50:
            obj.append(contour)
            added = True
            break
    if not added:
        objects.append([contour])

for obj in objects:
    x_min = min(obj, key=lambda x: x[0])[0]
    y_min = min(obj, key=lambda x: x[1])[1]
    x_max = max(obj, key=lambda x: x[0])[0]
    y_max = max(obj, key=lambda x: x[1])[1]

    image2[y_min:y_max, x_min] = [0, 255, 0]
    image2[y_min:y_max, x_max] = [0, 255, 0]
    image2[y_min, x_min:x_max] = [0, 255, 0]
    image2[y_max, x_min:x_max] = [0, 255, 0]

cv2.imwrite('dublin_bbox.jpg', image2)

largest_object = max(objects, key=len)

# Calculate bounding box for the largest object
x_min = min(largest_object, key=lambda x: x[0])[0] + 1
y_min = min(largest_object, key=lambda x: x[1])[1] + 1
x_max = max(largest_object, key=lambda x: x[0])[0]
y_max = max(largest_object, key=lambda x: x[1])[1]

cropped_image = image2[y_min:y_max, x_min:x_max]
image3 = np.copy(cropped_image)
cropped_image1 = image1[y_min:y_max, x_min:x_max]
cropped_image2 = image2[y_min:y_max, x_min:x_max]

difference_cropped = np.abs(cropped_image1.astype(int) - cropped_image2.astype(int)).astype(np.uint8)
gray_difference_cropped = np.dot(difference_cropped[..., :3], [0.2989, 0.5870, 0.1140])
threshold_cropped = (gray_difference_cropped > 7) * 255

differences = []
for y in range(threshold_cropped.shape[0]):
    for x in range(threshold_cropped.shape[1]):
        if threshold_cropped[y, x] == 255:
            differences.append((x, y))

mask = np.zeros_like(image3[:, :, 0], dtype=np.uint8)
for difference in differences:
    mask[difference[1], difference[0]] = 1

image3_rgba = np.dstack((image3, np.full(image3.shape[:2], 255)))
image3_rgba[mask != 1] = [0, 0, 0, 0]
cv2.imwrite('bbox_without_background.png', image3_rgba)

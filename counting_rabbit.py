import cv2
import numpy as np

# Function to compute the Intersection over Union (IoU) between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute the coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Compute the area of the intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the union area
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    if union_area == 0:
        return 0  # Avoid division by zero
    return inter_area / union_area

# Load the input images in color (RGB)
image = cv2.imread('Counting/rabbit.jpeg')
template = cv2.imread('Counting/objects/rabbit_eye.jpeg')

# Set the scaling factor if necessary (here it's 1)
scale_factor = 1
new_width = int(template.shape[1] * scale_factor)
new_height = int(template.shape[0] * scale_factor)
template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Create a binary mask for the template in RGB
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(template_gray, 240, 255, cv2.THRESH_BINARY_INV)
mask_rgb = cv2.merge([mask, mask, mask])  # Replicate the mask for all 3 channels (RGB)

# Get template dimensions
template_h, template_w = template.shape[:2]

# Split the image and template into their R, G, and B channels
image_r, image_g, image_b = cv2.split(image)
template_r, template_g, template_b = cv2.split(template)
mask_r, mask_g, mask_b = cv2.split(mask_rgb)

# Perform template matching on each RGB channel using the corresponding mask
result_r = cv2.matchTemplate(image_r, template_r, cv2.TM_SQDIFF_NORMED, mask=mask_r)
result_g = cv2.matchTemplate(image_g, template_g, cv2.TM_SQDIFF_NORMED, mask=mask_g)
result_b = cv2.matchTemplate(image_b, template_b, cv2.TM_SQDIFF_NORMED, mask=mask_b)

# Combine the results (you can average them or take the maximum or minimum depending on the use case)
result = (result_r + result_g + result_b) / 3

# Set a threshold for considering a match as "good enough"
threshold = 0.02  

# Find all locations where the match result is below the threshold
locations = np.where(result <= threshold)

# Create a list of bounding boxes for all matches
bounding_boxes = []
for pt in zip(*locations[::-1]):  
    top_left = pt
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    bounding_boxes.append((top_left[0], top_left[1], template_w, template_h))

# Merge overlapping bounding boxes
merged_boxes = []
iou_threshold = 0.1  # IoU threshold to consider two boxes as overlapping
for box in bounding_boxes:
    should_merge = False
    for merged_box in merged_boxes:
        if compute_iou(box, merged_box) > iou_threshold:
            should_merge = True
            break
    if not should_merge:
        merged_boxes.append(box)

# Draw rectangles around all merged matches
image_copy = image.copy()
for box in merged_boxes:
    top_left = (box[0], box[1])
    bottom_right = (box[0] + box[2], box[1] + box[3])
    cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)

# Print the number of merged matches found
print(f"Number of matches found (after merging overlapping boxes): {len(merged_boxes)}")

# Make the window resizable
cv2.namedWindow("Matched Templates", cv2.WINDOW_NORMAL)

# Display the original image with all matches highlighted
cv2.imshow("Matched Templates", image_copy)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

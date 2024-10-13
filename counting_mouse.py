import cv2
import numpy as np

# Function to check if two bounding boxes are in each other's vicinity
def are_boxes_near(box1, box2, vicinity_threshold):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Compute the center coordinates of both boxes
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2

    # Compute the distance between the centers
    distance = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    return distance <= vicinity_threshold

# Load the input images in color (RGB)
image = cv2.imread('Counting/mouse.jpg')
template = cv2.imread('Counting/objects/mouse_fur.jpg')

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

threshold = 0.01  # Adjust this threshold based on the results

# Find all locations where the match result is below the threshold
locations = np.where(result <= threshold)

# Create a list of bounding boxes for all matches
bounding_boxes = []
for pt in zip(*locations[::-1]):  # Switch the order of locations (x, y)
    top_left = pt
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    bounding_boxes.append((top_left[0], top_left[1], template_w, template_h))

# Group bounding boxes that are near each other
vicinity_threshold = 50  # Define the vicinity threshold in pixels
grouped_boxes = []
for box in bounding_boxes:
    found_group = False
    for group in grouped_boxes:
        if any(are_boxes_near(box, other_box, vicinity_threshold) for other_box in group):
            group.append(box)
            found_group = True
            break
    if not found_group:
        grouped_boxes.append([box])

# Merge each group of bounding boxes into a single box
merged_boxes = []
for group in grouped_boxes:
    if len(group) == 1:
        merged_boxes.append(group[0])
    else:
        x_coords = [b[0] for b in group]
        y_coords = [b[1] for b in group]
        widths = [b[2] for b in group]
        heights = [b[3] for b in group]
        
        # Compute the merged bounding box around all boxes in the group
        top_left_x = min(x_coords)
        top_left_y = min(y_coords)
        bottom_right_x = max([x + w for x, w in zip(x_coords, widths)])
        bottom_right_y = max([y + h for y, h in zip(y_coords, heights)])
        
        merged_width = bottom_right_x - top_left_x
        merged_height = bottom_right_y - top_left_y
        
        merged_boxes.append((top_left_x, top_left_y, merged_width, merged_height))

# Draw rectangles around all merged matches
image_copy = image.copy()
for box in merged_boxes:
    top_left = (box[0], box[1])
    bottom_right = (box[0] + box[2], box[1] + box[3])
    cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)

# Print the number of merged matches found
print(f"Number of matches found (after grouping nearby boxes): {len(merged_boxes)}")

# Make the window resizable
cv2.namedWindow("Matched Templates", cv2.WINDOW_NORMAL)

# Display the original image with all matches highlighted
cv2.imshow("Matched Templates", image_copy)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

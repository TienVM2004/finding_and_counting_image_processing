import cv2
import numpy as np

"""
12 OBJECTS 2.jpg

ball 0.79
bone 0.67
bow 0.78
bug 0.78
butterfly 0.78
cheese
chicken
cloud
icecream 0.78
pizza
sausage
strawberry
"""
# Load the input images
image = cv2.imread('Finding/2_cropped.jpg')
template = cv2.imread('Finding/2_objects/chicken.jpg')

# Convert the original image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the template to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Create a binary mask for the template to highlight non-background areas
# Assuming the background is mostly white, we'll create a mask to filter it out
_, mask = cv2.threshold(template_gray, 240, 255, cv2.THRESH_BINARY_INV)

# Initialize variables to store the best match
best_scale = None
best_min_val = float('inf')
best_top_left = None
best_template_w, best_template_h = None, None

# Loop over different scales using numpy.linspace
for scale_factor in np.linspace(0.5, 1, 50):  # Scale factors from 50% to 100%
    # Resize the template
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    resized_template = cv2.resize(template_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Perform template matching using TM_SQDIFF_NORMED with a mask
    try:
        result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_SQDIFF_NORMED, mask=resized_mask)
    except cv2.error as e:
        print("Error during template matching:", e)
        continue
    
    # Find the best match position (minimum value for SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Check if the current match is better than the previous best match
    if min_val < best_min_val:
        best_min_val = min_val
        best_scale = scale_factor
        best_top_left = min_loc
        best_template_w, best_template_h = resized_template.shape[:2]

# Draw the rectangle around the best match on the original image
if best_top_left is not None:
    top_left = best_top_left
    bottom_right = (top_left[0] + best_template_w, top_left[1] + best_template_h)
    image_copy = image.copy()
    cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)

    # Make the window resizable
    cv2.namedWindow("Best Matched Template", cv2.WINDOW_NORMAL)

    # Display the image with the best match
    cv2.imshow("Best Matched Template", image_copy)
    print(f"Best scale factor: {best_scale}")
    print(f"Best matching value (min_val): {best_min_val}")

    # Wait for a key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No match found.")

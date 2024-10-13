import cv2
import numpy as np

"""
15 OBJECTS 1.jpg

ball 0.6
bear 0.63
bow 0.5
bunny 0.7
cake 0.6
car 0.7
duck 0.72
grape 0.72
horse 0.6
icecream 0.7
strawberry 0.6
watermelon 0.5
wood 0.7
"""

"""
12 OBJECTS 2.jpg

ball 0.79
bone 0.67
bow 0.78
bug 0.78
butterfly 0.78
cheese 0.78
chicken 0.71
cloud 0.78
icecream 0.78
pizza 0.78
sausage 0.78
strawberry 0.78
"""
# CHANGE THIS LINE ACCORDING TO THE OBJECTS YOU WANT TO FIND
SCALE_FACTOR = 1

# Load the input images
image = cv2.imread('Counting/pair_boot.jpg')
template = cv2.imread('Counting/objects/pair_boot.jpg')

new_width = int(template.shape[1] * SCALE_FACTOR)
new_height = int(template.shape[0] * SCALE_FACTOR)
template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Convert the template to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Create a binary mask for the template to highlight non-background areas
_, mask = cv2.threshold(template_gray, 240, 255, cv2.THRESH_BINARY_INV)

# Convert the original image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get template dimensions
template_h, template_w = template_gray.shape[:2]

# Perform template matching using TM_SQDIFF_NORMED with a mask
try:
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=mask)
except cv2.error as e:
    print("Error during template matching:", e)
    exit()

cv2.namedWindow("mask", cv2.WINDOW_NORMAL)    
cv2.imshow("mask", mask)

# Find the best match position (minimum value for SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw the rectangle around the best match
top_left = min_loc
bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
image_copy = image.copy()
cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)

# Make the window resizable
cv2.namedWindow("Matched Template", cv2.WINDOW_NORMAL)

# Display the original image (which you can resize manually by dragging the window)
cv2.imshow("Matched Template", image_copy)

# Wait for a key press
cv2.waitKey(0)
cv2.destroyAllWindows()

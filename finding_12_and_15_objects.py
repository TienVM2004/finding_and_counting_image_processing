import cv2
import numpy as np

# List of objects with their corresponding scale factors
objects_1 = {
    "ball": 0.6,
    "bear": 0.63,
    "bow": 0.5,
    "bunny": 0.7,
    "cake": 0.6,
    "car": 0.7,
    "duck": 0.72,
    "grape": 0.72,
    "horse": 0.6,
    "icecream": 0.7,
    "strawberry": 0.6,
    "watermelon": 0.5,
    "wood": 0.7
}
objects_2 = {
    "ball": 0.79,
    "bone": 0.67,
    "bow": 0.78,
    "bug": 0.78,
    "butterfly": 0.78,
    "cheese": 0.78,
    "chicken": 0.71,
    "cloud": 0.78,
    "icecream": 0.78,
    "pizza": 0.78,
    "sausage": 0.78,
    "strawberry": 0.78
}
# Load the source image
image = cv2.imread('Finding/2_cropped.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Loop through each object and its corresponding scale factor
for object_name, scale_factor in objects_2.items():
    # Load the template image for the current object
    template = cv2.imread(f'Finding/2_objects/{object_name}.jpg')

    if template is None:
        print(f"Template for {object_name} not found!")
        continue

    # Resize the template based on the scale factor
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Convert the template to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Create a binary mask for the template to highlight non-background areas
    _, mask = cv2.threshold(template_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Get template dimensions
    template_h, template_w = template_gray.shape[:2]

    # Perform template matching using TM_SQDIFF_NORMED with a mask
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_SQDIFF_NORMED, mask=mask)

    # Find the best match position (minimum value for SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw the rectangle around the best match
    top_left = min_loc
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Add the object's name above the rectangle
    cv2.putText(image, object_name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Make the window resizable and display the final result
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

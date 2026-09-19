from pathlib import Path

import cv2 as cv
import numpy as np


# --------------------------------
# SETTINGS
# --------------------------------

# Number of color clusters K-Means should create
NUM_COLORS = 6

# Minimum object area in pixels
MIN_AREA = 300

# HSV tolerance
TOLERANCE_H = 12
TOLERANCE_S = 70
TOLERANCE_V = 70


# --------------------------------
# FUNCTION: GIVE COLOR A NAME
# --------------------------------

def get_color_name(hue, saturation, value):

    # Very dark
    if value < 50:
        return "Black"

    # Low saturation = white / gray
    if saturation < 40:

        if value > 200:
            return "White"

        return "Gray"

    # Colored pixels
    if hue < 10 or hue >= 170:
        return "Red"

    elif hue < 22:
        return "Orange"

    elif hue < 35:
        return "Yellow"

    elif hue < 85:
        return "Green"

    elif hue < 100:
        return "Cyan"

    elif hue < 130:
        return "Blue"

    elif hue < 150:
        return "Purple"

    elif hue < 170:
        return "Pink"

    return "Unknown"


# --------------------------------
# 1. LOAD IMAGE
# --------------------------------

img_path = Path(__file__).resolve().parent / "OIP.webp"

img = cv.imread(str(img_path))

if img is None:
    raise FileNotFoundError(
        f"Could not load image: {img_path}"
    )


# Make a copy for drawing results
result_img = img.copy()


# --------------------------------
# 2. CONVERT BGR → HSV
# --------------------------------

hsv = cv.cvtColor(
    img,
    cv.COLOR_BGR2HSV
)


# --------------------------------
# 3. K-MEANS COLOR CLUSTERING
# --------------------------------

# Convert image into a list of pixels
pixels = img.reshape((-1, 3))

pixels = np.float32(pixels)


# K-Means stopping criteria
criteria = (
    cv.TERM_CRITERIA_EPS +
    cv.TERM_CRITERIA_MAX_ITER,
    10,
    1.0
)


# Cluster pixels into NUM_COLORS groups
_, labels, centers = cv.kmeans(
    pixels,
    NUM_COLORS,
    None,
    criteria,
    10,
    cv.KMEANS_RANDOM_CENTERS
)


# Convert cluster centers to integer BGR
dominant_colors_bgr = np.uint8(centers)


print(
    f"Detected {NUM_COLORS} color cluster(s):"
)


# --------------------------------
# 4. DISPLAY DISCOVERED COLORS
# --------------------------------

for i, color_bgr in enumerate(
    dominant_colors_bgr
):

    b = int(color_bgr[0])
    g = int(color_bgr[1])
    r = int(color_bgr[2])

    print(
        f"  Color {i + 1}: "
        f"BGR = ({b}, {g}, {r})"
    )

    print(
        f"            RGB = ({r}, {g}, {b})"
    )


# --------------------------------
# 5. CREATE MORPHOLOGY KERNEL
# --------------------------------

kernel = cv.getStructuringElement(
    cv.MORPH_RECT,
    (5, 5)
)


# --------------------------------
# 6. PROCESS EACH COLOR
# --------------------------------

for i, color_bgr in enumerate(
    dominant_colors_bgr
):

    # Get BGR values
    b = int(color_bgr[0])
    g = int(color_bgr[1])
    r = int(color_bgr[2])


    # --------------------------------
    # CONVERT THIS COLOR BGR → HSV
    # --------------------------------

    color_bgr_array = np.uint8(
        [[color_bgr]]
    )

    color_hsv = cv.cvtColor(
        color_bgr_array,
        cv.COLOR_BGR2HSV
    )[0][0]


    hue = int(color_hsv[0])
    saturation = int(color_hsv[1])
    value = int(color_hsv[2])


    # --------------------------------
    # GIVE COLOR A HUMAN NAME
    # --------------------------------

    color_name = get_color_name(
        hue,
        saturation,
        value
    )


    print(
        f"\nCluster {i + 1}: "
        f"{color_name}"
    )

    print(
        f"  RGB: "
        f"({r}, {g}, {b})"
    )

    print(
        f"  HSV: "
        f"({hue}, {saturation}, {value})"
    )


    # --------------------------------
    # CREATE HSV RANGE
    # --------------------------------

    lower_hue = max(
        hue - TOLERANCE_H,
        0
    )

    upper_hue = min(
        hue + TOLERANCE_H,
        179
    )


    lower_saturation = max(
        saturation - TOLERANCE_S,
        0
    )

    upper_saturation = min(
        saturation + TOLERANCE_S,
        255
    )


    lower_value = max(
        value - TOLERANCE_V,
        0
    )

    upper_value = min(
        value + TOLERANCE_V,
        255
    )


    lower_hsv = (
        lower_hue,
        lower_saturation,
        lower_value
    )

    upper_hsv = (
        upper_hue,
        upper_saturation,
        upper_value
    )


    # --------------------------------
    # CREATE MASK
    # --------------------------------

    mask = cv.inRange(
        hsv,
        lower_hsv,
        upper_hsv
    )


    # --------------------------------
    # CLEAN MASK
    # --------------------------------

    clean_mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=1
    )

    clean_mask = cv.morphologyEx(
        clean_mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=2
    )


    # --------------------------------
    # FIND CONTOURS
    # --------------------------------

    contours, _ = cv.findContours(
        clean_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------
    # FIND OBJECTS
    # --------------------------------

    for contour in contours:

        area = cv.contourArea(
            contour
        )


        # Ignore tiny objects
        if area > MIN_AREA:

            # Get bounding box
            x, y, width, height = (
                cv.boundingRect(contour)
            )


            # --------------------------------
            # DRAW BOUNDING BOX
            # --------------------------------

            cv.rectangle(
                result_img,
                (x, y),
                (x + width, y + height),
                (b, g, r),
                3
            )


            # --------------------------------
            # CREATE LABEL
            # --------------------------------

            label = (
                f"{color_name} "
                f"RGB({r},{g},{b})"
            )


            # --------------------------------
            # DRAW LABEL
            # --------------------------------

            cv.putText(
                result_img,
                label,
                (x, y - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (b, g, r),
                2
            )


            # --------------------------------
            # PRINT OBJECT INFORMATION
            # --------------------------------

            print(
                f"\nObject detected!"
            )

            print(
                f"  Color   : {color_name}"
            )

            print(
                f"  RGB     : "
                f"({r}, {g}, {b})"
            )

            print(
                f"  HSV     : "
                f"({hue}, {saturation}, {value})"
            )

            print(
                f"  Area    : "
                f"{int(area)} px"
            )

            print(
                f"  Position: "
                f"X={x}, Y={y}"
            )

            print(
                f"  Size    : "
                f"{width} x {height}"
            )


    # --------------------------------
    # SHOW MASK
    # --------------------------------

    cv.imshow(
        f"Mask - {color_name}",
        clean_mask
    )


# --------------------------------
# 7. SHOW FINAL RESULTS
# --------------------------------

cv.imshow(
    "Original Image",
    img
)

cv.imshow(
    "All Colors - Detected Objects",
    result_img
)


# --------------------------------
# 8. WAIT AND CLOSE
# --------------------------------

print("\n" + "-" * 50)

print(
    "Press any key on any window "
    "to close everything."
)

print("-" * 50)


cv.waitKey(0)

cv.destroyAllWindows()
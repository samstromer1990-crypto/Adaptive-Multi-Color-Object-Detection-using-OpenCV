import cv2 as cv
import numpy as np


# ============================================
# SETTINGS
# ============================================

NUM_COLORS = 6

MIN_AREA = 500

TOLERANCE_H = 12
TOLERANCE_S = 70
TOLERANCE_V = 70

# Run K-Means again after this many frames
KMEANS_INTERVAL = 60


# ============================================
# FUNCTION: GIVE COLOR A NAME
# ============================================

def get_color_name(hue, saturation, value):

    # Very dark
    if value < 50:
        return "Black"

    # Low saturation
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


# ============================================
# FUNCTION: FIND DOMINANT COLORS
# ============================================

def find_dominant_colors(frame):

    # Resize frame for faster K-Means
    small = cv.resize(
        frame,
        (320, 240)
    )

    # Convert image into list of pixels
    pixels = small.reshape((-1, 3))

    # K-Means needs float32
    pixels = np.float32(pixels)

    # K-Means stopping criteria
    criteria = (
        cv.TERM_CRITERIA_EPS +
        cv.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    # Run K-Means
    _, labels, centers = cv.kmeans(
        pixels,
        NUM_COLORS,
        None,
        criteria,
        5,
        cv.KMEANS_PP_CENTERS
    )

    # Convert cluster centers back to uint8
    centers = np.uint8(centers)

    return centers


# ============================================
# FUNCTION: CREATE COLOR INFORMATION
# ============================================

def create_color_information(centers):

    color_information = []

    for color_bgr in centers:

        b = int(color_bgr[0])
        g = int(color_bgr[1])
        r = int(color_bgr[2])

        # Convert BGR color into HSV
        color_pixel = np.uint8(
            [[color_bgr]]
        )

        color_hsv = cv.cvtColor(
            color_pixel,
            cv.COLOR_BGR2HSV
        )[0][0]

        hue = int(color_hsv[0])
        saturation = int(color_hsv[1])
        value = int(color_hsv[2])

        # Give human-readable name
        color_name = get_color_name(
            hue,
            saturation,
            value
        )

        color_information.append({
            "name": color_name,
            "bgr": (b, g, r),
            "hsv": (hue, saturation, value)
        })

    return color_information


# ============================================
# OPEN WEBCAM
# ============================================

cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam"
    )


# ============================================
# VARIABLES
# ============================================

frame_count = 0

color_information = []


# ============================================
# MAIN LOOP
# ============================================

while True:

    # ----------------------------------------
    # GET FRAME
    # ----------------------------------------

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read frame"
        )

        break


    # ----------------------------------------
    # RUN K-MEANS PERIODICALLY
    # ----------------------------------------

    if (
        frame_count % KMEANS_INTERVAL == 0
        or len(color_information) == 0
    ):

        centers = find_dominant_colors(
            frame
        )

        color_information = (
            create_color_information(
                centers
            )
        )

        print(
            "\nDetected dominant colors:"
        )

        for color in color_information:

            print(
                f"{color['name']} "
                f"RGB="
                f"({color['bgr'][2]}, "
                f"{color['bgr'][1]}, "
                f"{color['bgr'][0]}) "
                f"HSV="
                f"{color['hsv']}"
            )


    # ----------------------------------------
    # CONVERT FRAME TO HSV
    # ----------------------------------------

    hsv = cv.cvtColor(
        frame,
        cv.COLOR_BGR2HSV
    )


    # ----------------------------------------
    # CREATE RESULT IMAGE
    # ----------------------------------------

    result = frame.copy()


    # ========================================
    # DETECT EACH DOMINANT COLOR
    # ========================================

    for color in color_information:

        color_name = color["name"]

        b, g, r = color["bgr"]

        hue, saturation, value = color["hsv"]


        # ------------------------------------
        # CREATE HSV RANGE
        # ------------------------------------

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


        # ------------------------------------
        # CREATE MASK
        # ------------------------------------

        mask = cv.inRange(
            hsv,
            lower_hsv,
            upper_hsv
        )


        # ------------------------------------
        # CLEAN MASK
        # ------------------------------------

        kernel = cv.getStructuringElement(
            cv.MORPH_RECT,
            (5, 5)
        )

        clean_mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            kernel
        )

        clean_mask = cv.morphologyEx(
            clean_mask,
            cv.MORPH_CLOSE,
            kernel
        )


        # ------------------------------------
        # FIND CONTOURS
        # ------------------------------------

        contours, _ = cv.findContours(
            clean_mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )


        # ------------------------------------
        # DETECT OBJECTS
        # ------------------------------------

        for contour in contours:

            area = cv.contourArea(
                contour
            )

            if area > MIN_AREA:

                x, y, width, height = (
                    cv.boundingRect(contour)
                )


                # Draw bounding box
                cv.rectangle(
                    result,
                    (x, y),
                    (x + width, y + height),
                    (b, g, r),
                    2
                )


                # Create label
                label = (
                    f"{color_name}"
                )


                # Draw label
                cv.putText(
                    result,
                    label,
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (b, g, r),
                    2
                )


    # ========================================
    # SHOW RESULTS
    # ========================================

    cv.imshow(
        "Webcam - Color Object Detector",
        result
    )


    # ========================================
    # PRESS Q TO QUIT
    # ========================================

    if cv.waitKey(1) & 0xFF == ord('q'):

        break


    frame_count += 1


# ============================================
# RELEASE WEBCAM
# ============================================

cap.release()

cv.destroyAllWindows()
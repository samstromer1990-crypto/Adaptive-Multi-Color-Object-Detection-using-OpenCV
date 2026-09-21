import cv2 as cv


cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break


    #  BGR → HSV
    

    hsv = cv.cvtColor(
        frame,
        cv.COLOR_BGR2HSV
    )


   
    # RED COLOR RANGE
   

    lower_red1 = (0, 50, 50)
    upper_red1 = (10, 255, 255)

    lower_red2 = (170, 50, 50)
    upper_red2 = (179, 255, 255)


    mask1 = cv.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    mask2 = cv.inRange(
        hsv,
        lower_red2,
        upper_red2
    )


    # Combine them
    mask = cv.bitwise_or(
        mask1,
        mask2
    )


    #  CLEAN MASK

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


    #  FIND CONTOURS

    contours, _ = cv.findContours(
        clean_mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )


    # FIND RED OBJECTS

    for contour in contours:

        area = cv.contourArea(contour)

        if area > 500:

            x, y, width, height = (
                cv.boundingRect(contour)
            )


            # Draw bounding box
            cv.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )


            # Add label
            cv.putText(
                frame,
                "RED OBJECT",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )




    cv.imshow(
        "Webcam - Red Object Detector",
        frame
    )

    cv.imshow(
        "Red Mask",
        clean_mask
    )



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------------
# 8. RELEASE WEBCAM
# --------------------------------

cap.release()
cv.destroyAllWindows()

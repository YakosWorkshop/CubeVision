import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Canny edge detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blurred, 20, 40)
    
    # Dilate the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv.dilate(canny, kernel, iterations=2)
    
    # Find contours
    (contours, hierarchy) = cv.findContours(dilated.copy(), 
                                            cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area < 1000: 
            continue
        # Approximate contour
        epsilon = 0.03 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # If shape has 4 corners → likely square
        if len(approx) == 4:
            cv.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

#def approxSquare(approx)
import cv2
import numpy as np

def nothing(x):
    pass

# Create window for sliders
cv2.namedWindow("Trackbars")

# HSV range sliders
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Val Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, nothing)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Denoise using Gaussian blur
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Read HSV slider positions
    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create mask and apply
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display windows
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


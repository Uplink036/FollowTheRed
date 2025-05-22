import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam.")

            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2

            # Get BGR value from original frame
            bgr = frame[center_y, center_x]

            # Convert frame to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = hsv_frame[center_y, center_x]

            print(f"Center Pixel BGR: {tuple(bgr)} | HSV: {tuple(hsv)}")

            # Draw marker and overlay text
            # cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"BGR: {tuple(bgr)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f"HSV: {tuple(hsv)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


import cv2
import numpy as np

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Define color ranges in HSV format
colors = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Green": ([40, 40, 40], [80, 255, 255]),
    "Blue": ([90, 50, 50], [130, 255, 255]),
    "Yellow": ([20, 100, 100], [40, 255, 255])
}

while True:
    # Read a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in colors.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)

        # Create a mask for the color
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Find contours of detected colors
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 4000:  # Threshold to ignore small noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, f"{color_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show original video frame with detected color boxes
    cv2.imshow("Webcam - Color Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
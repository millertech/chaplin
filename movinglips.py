import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    cap.release()
    exit(1)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

mouth_open = False

# For live graph
window_size = 200
circle_count_history = deque(maxlen=window_size)
radius_history = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Mouth Circles')
line2, = ax.plot([], [], label='Mean Radius')
ax.set_ylim(0, 50)  # Adjust as needed for your expected radius range
ax.set_xlim(0, window_size)
ax.legend()
ax.set_title("Detected Circles and Mean Radius in Mouth ROI (Live)")
ax.set_xlabel("Frame")
ax.set_ylabel("Value")

p_x, p_y, p_w, p_h = 0, 0, 0, 0
box_alpha = 0.01

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        y = int(box_alpha * p_y + (1 - box_alpha) * y)
        x = int(box_alpha * p_x + (1 - box_alpha) * x)
        w = int(box_alpha * p_w + (1 - box_alpha) * w)
        h = int(box_alpha * p_h + (1 - box_alpha) * h)
        p_x, p_y, p_w, p_h = x, y, w, h

        # Focus on mouth area
        y_mouth = int(y + h * 0.6)
        h_mouth = int(h * 0.5)
        x_mouth = int(x + w * 0.15)
        w_mouth = int(w * 0.7)

        mouth_roi = gray[y_mouth:y_mouth + h_mouth, x_mouth:x_mouth + w_mouth]
        mouth_roi_blur = cv2.medianBlur(mouth_roi, 7)

        # Preprocess mouth ROI for better circle detection
        mouth_roi_blur = cv2.GaussianBlur(mouth_roi, (9, 9), 2)
        mouth_roi_edges = cv2.Canny(mouth_roi_blur, 50, 150)

        # Hough Circle detection (stricter parameters)
        circles = cv2.HoughCircles(
            mouth_roi_edges,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=int(h_mouth * 0.5),
            param1=100,
            param2=35,  # Higher threshold for stricter detection
            minRadius=int(h_mouth * 0.05),
            maxRadius=int(h_mouth * 0.45)
        )

        circle_count = 0
        mean_radius = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Optionally, filter circles by position (centered in ROI)
            filtered_circles = []
            radii = []
            for i in circles[0, :]:
                cx, cy, r = i
                if (cx > w_mouth * 0.25 and cx < w_mouth * 0.75 and
                    cy > h_mouth * 0.25 and cy < h_mouth * 0.75):
                    filtered_circles.append(i)
                    radii.append(r)
                    cv2.circle(frame, (x_mouth + cx, y_mouth + cy), r, (0, 255, 0), 2)
                    cv2.circle(frame, (x_mouth + cx, y_mouth + cy), 2, (0, 0, 255), 3)
            circle_count = len(filtered_circles)
            if radii:
                mean_radius = np.mean(radii)
        circle_count_history.append(circle_count)
        radius_history.append(mean_radius)

        # Threshold: if at least one circle is detected, consider mouth open
        if circle_count > 0:
            mouth_open = True
            cv2.putText(frame, f"Mouth Open! Circles: {circle_count} MeanR: {mean_radius:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        else:
            mouth_open = False

        # Draw rectangle around mouth ROI
        cv2.rectangle(frame, (x_mouth, y_mouth), (x_mouth + w_mouth, y_mouth + h_mouth), (255, 0, 0), 2)

    # Update live graph
    line1.set_data(range(len(circle_count_history)), list(circle_count_history))
    line2.set_data(range(len(radius_history)), list(radius_history))
    ax.set_xlim(max(0, len(circle_count_history) - window_size), len(circle_count_history))
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

    cv2.imshow('Mouth Open Detection (Circle)', frame)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
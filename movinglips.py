import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

mouth_moving = False

# Low-pass filter setup
alpha = 0.5  # Smoothing factor (0 < alpha < 1), lower = smoother
filtered_movement_diff = 0.0
mean_face_movement_filtered = 0.0
mean_mouth_movement_filtered = 0.0
p_x, p_y = 0, 0
p_h, p_w = 0, 0
box_alpha = 0.01

# For live graph
window_size = 200
mouth_history = deque(maxlen=window_size)
face_history = deque(maxlen=window_size)
diff_history = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Mouth')
line2, = ax.plot([], [], label='Face')
line3, = ax.plot([], [], label='Diff')
ax.set_ylim(0, 5)
ax.set_xlim(0, window_size)
ax.legend()
ax.set_title("Mean Filtered Movement (Live)")
ax.set_xlabel("Frame")
ax.set_ylabel("Movement")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)#what are the arguments here? 
    #scaleFactor=1.3, minNeighbors=5
    #scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
    #minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    #adjust scalefactor when you want to detect smaller or larger faces, a larger value = less detections
    # a smaller value = more detections, detections are more accurate with smaller values but slower

    #minNeighbors: higher value results in less detections but with higher quality
    for (x, y, w, h) in faces:
        y = int(box_alpha * p_y + (1 - box_alpha) * y)
        x = int(box_alpha * p_x + (1 - box_alpha) * x)
        w = int(box_alpha * p_w + (1 - box_alpha) * w)
        h = int(box_alpha * p_h + (1 - box_alpha) * h)
        p_x, p_y, p_w, p_h = x, y, w, h
        #make the face box smaller to just include mouth area
        y = int(y + h * 0.4)#adjust y to start halfway down the face
        h = int(h * 0.55)#adjust height to be half the face height
        x = int(x + w * 0.15)#adjust x to start a bit in from the left
        w = int(w * 0.7)#adjust width to be narrower

        # Face ROI
        face_roi = gray[y:y+h, x:x+w]
        prev_face_roi = prev_gray[y:y+h, x:x+w]

        # Mouth ROI (lower part of the face)
        mouth_roi = gray[y + h//2:y + h, x:x + w]
        prev_mouth_roi = prev_gray[y + h//2:y + h, x:x + w]

        # Optical flow for face and mouth
        flow_face = cv2.calcOpticalFlowFarneback(prev_face_roi, face_roi, None,
                                                 pyr_scale=0.5, levels=3, winsize=15,
                                                 iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag_face, _ = cv2.cartToPolar(flow_face[..., 0], flow_face[..., 1])
        mean_face_movement = np.mean(mag_face)

        flow_mouth = cv2.calcOpticalFlowFarneback(prev_mouth_roi, mouth_roi, None,
                                                  pyr_scale=0.5, levels=3, winsize=15,
                                                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag_mouth, _ = cv2.cartToPolar(flow_mouth[..., 0], flow_mouth[..., 1])
        mean_mouth_movement = np.mean(mag_mouth)

        # Only trigger if mouth moves significantly more than face
        mean_face_movement_filtered = round(alpha * mean_face_movement + (1 - alpha) * mean_face_movement_filtered, ndigits=1)
        mean_mouth_movement_filtered = round(alpha * mean_mouth_movement + (1 - alpha) * mean_mouth_movement_filtered, ndigits=1)

        movement_diff = mean_mouth_movement - mean_face_movement

        # Low-pass filter: smooth the movement_diff
        filtered_movement_diff = mean_mouth_movement_filtered - mean_face_movement_filtered

        # Store for live graph
        mouth_history.append(mean_mouth_movement_filtered)
        face_history.append(mean_face_movement_filtered)
        diff_history.append(filtered_movement_diff)

        # Thresholds (tune as needed)
        print(f'D:{movement_diff:.2f}  \tFD:{filtered_movement_diff:.2f}  \tM:{mean_mouth_movement_filtered:.2f}  \tF:{mean_face_movement_filtered:.2f}')
        if  mean_mouth_movement_filtered > 1.0 and mean_face_movement_filtered < 1.0:
            mouth_moving = True
            print(f"Mouth is moving: FD:{filtered_movement_diff:.2f}  M:{mean_mouth_movement_filtered:.2f}  F:{mean_face_movement_filtered:.2f}")
            cv2.putText(frame, f"Lips Moving! fdiff:{filtered_movement_diff:.2f} mouth: {mean_mouth_movement_filtered:.2f} face: {mean_face_movement_filtered:.2f} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        else:
            mouth_moving = False

        cv2.rectangle(frame, (x, y + h//2), (x + w, y + h), (255, 0, 0), 2)

    # Update live graph
    line1.set_data(range(len(mouth_history)), list(mouth_history))
    line2.set_data(range(len(face_history)), list(face_history))
    line3.set_data(range(len(diff_history)), list(diff_history))
    ax.set_xlim(max(0, len(mouth_history) - window_size), len(mouth_history))
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

    cv2.imshow('Lip Movement Detection', frame)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
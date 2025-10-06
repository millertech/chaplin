import cv2
import numpy as np

# Load face detector (Haar cascade for simplicity)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

mouth_moving = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Define mouth ROI (lower part of the face)
        mouth_roi = gray[y + h//2:y + h, x:x + w]
        prev_mouth_roi = prev_gray[y + h//2:y + h, x:x + w]

        # Calculate optical flow in mouth ROI
        flow = cv2.calcOpticalFlowFarneback(prev_mouth_roi, mouth_roi, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_movement = np.mean(mag)

        # Threshold for movement detection (tune as needed)
        if mean_movement > 1.5:
            mouth_moving = True
            cv2.putText(frame, "Lips Moving!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        else:
            mouth_moving = False

        # Draw rectangle around mouth
        cv2.rectangle(frame, (x, y + h//2), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Lip Movement Detection', frame)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
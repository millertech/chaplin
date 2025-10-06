import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

mouth_moving = False

# Low-pass filter setup
alpha = 0.35  # Smoothing factor (0 < alpha < 1), lower = smoother
filtered_movement_diff = 0.0
mean_face_movement_filtered = 0.0
mean_mouth_movement_filtered = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
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
        mean_face_movement_filtered = alpha * mean_face_movement + (1 - alpha) * mean_face_movement_filtered
        mean_mouth_movement_filtered = alpha * mean_mouth_movement + (1 - alpha) * mean_mouth_movement_filtered

        movement_diff = mean_mouth_movement - mean_face_movement

        # Low-pass filter: smooth the movement_diff
        filtered_movement_diff = mean_mouth_movement_filtered - mean_face_movement_filtered

        # Thresholds (tune as needed)
        print(f'D:{movement_diff:.2f}  \tFD:{filtered_movement_diff:.2f}  \tM:{mean_mouth_movement_filtered:.2f}  \tF:{mean_face_movement_filtered:.2f}')
        if  mean_mouth_movement_filtered > 1.0 and mean_face_movement_filtered < 1.0:#filtered_movement_diff > 1.1 and
            mouth_moving = True
            print(f"Mouth is moving: FD:{filtered_movement_diff:.2f}  M:{mean_mouth_movement_filtered:.2f}  F:{mean_face_movement_filtered:.2f}")
            cv2.putText(frame, f"Lips Moving! fdiff:{filtered_movement_diff:.2f} mouth: {mean_mouth_movement_filtered:.2f} face: {mean_face_movement_filtered:.2f} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        else:
            mouth_moving = False

        cv2.rectangle(frame, (x, y + h//2), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Lip Movement Detection', frame)
    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
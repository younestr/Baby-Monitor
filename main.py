#!/usr/bin/python3

import cv2
from picamera2 import Picamera2

# Grab images as numpy arrays and leave everything else to OpenCV.
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))  # Adjust resolution
picam2.start()

# Initialize variables for motion detection
prev_frame = None

while True:
    im = picam2.capture_array()

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))

    # Perform motion detection
    motion_detected = False
    if prev_frame is not None:
        # Convert the current frame to grayscale
        current_frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_frame, current_frame_gray)

        # Threshold the difference image
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if motion is detected
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Adjust threshold as needed
                motion_detected = True
                break

    # Update the previous frame
    prev_frame = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Display text indicating motion detection status
    if motion_detected:
        cv2.putText(im, 'Motion Detected', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else:
        cv2.putText(im, 'No Motion', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Camera", im)
    cv2.waitKey(1)


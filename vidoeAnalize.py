import cv2
import numpy as np

vid = cv2.VideoCapture("video/video.mp4")
sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=110, detectShadows=True)

while True:
    ret, frame = vid.read()

    if not ret:
        break


    if frame.shape[0] < 512 or frame.shape[1] < 512:
        continue

    frame = cv2.resize(frame, (512, 512))
    uygula = sub.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("perception", uygula)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()

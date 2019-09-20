import cv2
import numpy as np
cap = cv2.VideoCapture(0)
i = 0
while (1):
    ret, frame = cap.read()
    frame = frame[240 - 128:240 + 128, 320 - 128:320 + 128, :]
    k = cv2.waitKey(1)
    if k == 27:
        break
    cv2.imwrite('/home/chen/Documents/Mobile_hand/src/snapshots_posenet/baseline/real/' + str(i) + '.jpg', frame)
    i += 1
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()
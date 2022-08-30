import hand_tracking_module as HTM
import cv2
import time


cap = cv2.VideoCapture(0)
# previous and current time
previous_time = 0
current_time = 0

detector = HTM.HandDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw = False)

    if len(lm_list) != 0:
        print(lm_list[0])

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    # write the fps to the screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
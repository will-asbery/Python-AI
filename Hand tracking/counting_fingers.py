import cv2
import time
import os
import hand_tracking_module as HTM

def get_fingers(landmark_list):
    fingers = []
    if len(landmark_list) != 0:
        for id in tip_ids:
            if id != 4:
                fingers.append(1 if landmark_list[id][2] < landmark_list[id - 2][2] else 0)
            else:
                # for thumb i need hand orientation
                if landmark_list[4][1] < landmark_list[20][1]:
                    # if the left hand is up
                    fingers.append(1 if landmark_list[id][1] < landmark_list[id-2][1] else 0)
                else:
                    # if the right hand is up
                    fingers.append(1 if landmark_list[id][1] > landmark_list[id-2][1] else 0)

    return fingers, fingers.count(1)

# initialising camera and features
cam_width, cam_height = 640, 480
capture = cv2.VideoCapture(0)
capture.set(3, cam_width)
capture.set(4, cam_height)

# create the hand detector from the imported module
detector = HTM.HandDetector(detection_confidence = 0.75)
# arr to hold all the tip of the fingers values
# goes from thumb to pinky
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = capture.read()
    img = detector.find_hands(img)

    landmark_list = detector.find_position(img, draw = False)
    fingers, finger_count = get_fingers(landmark_list)

    # when None returned
    if not finger_count:
        finger_count = 0
    else:
        if fingers == [0, 0, 1, 0, 0]:
            cv2.putText(img, f'Fuck you too', (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.putText(img, f'{finger_count}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
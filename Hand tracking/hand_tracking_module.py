import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode = False, max_hands = 2, model_complexity = 1, detection_confidence = 0.5, track_confidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, 
                                        self.model_complexity, self.detection_confidence, 
                                        self.track_confidence)

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw = True):
        ''' Function to find the hands on the screen '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmark, 
                                                self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_number = 0, draw = True):
        ''' Function to retun a list of all given hand point positions '''
        # list to hold each positions location
        lm_list = []
        # if hands are detected
        if self.results.multi_hand_landmarks:
            # only draw the hand which we are detecting
            my_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                # add to list for later use
                lm_list.append([id, center_x, center_y])
                # only draw if draw flag is set to true
                if draw:
                    cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    # previous and current time
    previous_time = 0
    current_time = 0

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[0])

        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time

        # write the fps to the screen
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
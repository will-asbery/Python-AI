import cv2
import mediapipe as mp

class FaceDetector():
    def __init__(self, min_detection_confidence = 0.5):
        self.min_detection_confidence = min_detection_confidence
        
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_confidence)

    def find_faces(self, img, draw = True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_RGB)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                conf = int(detection.score[0] * 100)

                bbox = detection.location_data.relative_bounding_box
                image_height, image_width, image_channel = img.shape
                bbox = int(bbox.xmin * image_width), int(bbox.ymin * image_height), \
                    int(bbox.width * image_width), int(bbox.height * image_height)

                bboxs.append([id, bbox, conf])
                img = self.fancy_draw(img, bbox, conf)

        return img, bboxs

    def fancy_draw(self, img, bbox, conf, length = 30, thickness = 10, rec_thickness = 1):
        x, y, width, height = bbox
        x1, y1 = x + width, y + height

        cv2.rectangle(img, bbox, (255, 0, 255), rec_thickness)
        cv2.line(img, (x, y), (x + length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y + height), (x + length, y + height), (255, 0, 255), thickness)
        cv2.line(img, (x1 - length, y), (x1, y), (255, 0, 255), thickness)
        cv2.line(img, (x1 - length, y + height), (x1, y + height), (255, 0, 255), thickness)

        cv2.line(img, (x, y), (x, y + length), (255, 0, 255), thickness) # top left
        cv2.line(img, (x, y1 - length), (x, y1), (255, 0, 255), thickness) # bottom left
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness) # top right
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness) # bottom right

        cv2.putText(img, f'{conf}%', (bbox[0] + 15, bbox[1] + 25), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1)

        return img

    

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.find_faces(img)

        cv2.imshow("Webcam 0", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
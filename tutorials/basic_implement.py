import cv2
import mediapipe as mp
import time

class BasicDetector:
    def __init__(self, mode=False, complexity=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.solution = mp.solutions.pose
        self.detector = self.solution.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            min_detection_confidence=detection_conf
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def process_image(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.detector.process(img_rgb)

        if draw and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.solution.POSE_CONNECTIONS
            )
        return img

def main():
    cap = cv2.VideoCapture(0)
    detector = BasicDetector()
    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.process_image(img)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Basic CV Template", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
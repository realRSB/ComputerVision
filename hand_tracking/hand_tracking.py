# imports
import cv2
import mediapipe as mp
import time


class HandDetector:
    """A class for detecting and tracking hands using MediaPipe."""
    
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        """Initialize the hand detector.
        
        Args:
            mode (bool): If True, treats each frame independently
            max_hands (int): Maximum number of hands to detect
            detection_conf (float): Minimum confidence for detection
            track_conf (float): Minimum confidence for tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Drawing specifications
        self.landmark_style = self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
        self.connection_style = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)

    def find_hands(self, img, draw=True):
        """Detect hands in an image.
        
        Args:
            img: Input image (BGR format)
            draw (bool): If True, draws hand landmarks
        
        Returns:
            Image with or without drawn landmarks
        """
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style,
                    self.connection_style
                )
        return img

    def find_positions(self, img, hand_no=0, draw=True):
        """Find hand landmark positions.
        
        Args:
            img: Input image
            hand_no (int): Which hand to track (0 is first)
            draw (bool): If True, draws position markers
        
        Returns:
            List of landmark positions [id, x, y]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            target_hand = self.results.multi_hand_landmarks[hand_no]
            
            for idx, landmark in enumerate(target_hand.landmark):
                height, width, _ = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([idx, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    
        return landmark_list


def main():
    """Main function demonstrating hand tracking capabilities."""
    
    # Initialize camera and detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    prev_time = 0
    
    # Landmark index reference
    """
    Hand Landmark indices:
    0=WRIST, 1=THUMB_CMC, 2=THUMB_MCP, 3=THUMB_IP, 4=THUMB_TIP,
    5=INDEX_FINGER_MCP, 6=INDEX_FINGER_PIP, 7=INDEX_FINGER_DIP, 8=INDEX_FINGER_TIP,
    9=MIDDLE_FINGER_MCP, 10=MIDDLE_FINGER_PIP, 11=MIDDLE_FINGER_DIP, 12=MIDDLE_FINGER_TIP,
    13=RING_FINGER_MCP, 14=RING_FINGER_PIP, 15=RING_FINGER_DIP, 16=RING_FINGER_TIP,
    17=PINKY_MCP, 18=PINKY_PIP, 19=PINKY_DIP, 20=PINKY_TIP
    """
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        img = detector.find_hands(img)
        landmark_list = detector.find_positions(img)
        
        if landmark_list:
            print(f"Thumb tip position: {landmark_list[4]}")
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracking", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
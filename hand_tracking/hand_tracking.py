"""
Hand Tracking Module
------------------
This module provides real-time hand detection and tracking capabilities using MediaPipe.
It can detect multiple hands and track 21 landmarks per hand.

Key Features:
- Real-time hand detection
- Multi-hand tracking (up to 2 hands)
- 21 landmark points per hand
- FPS monitoring
- Customizable visualization
"""

import cv2
import mediapipe as mp
import time


# Define hand landmark indices for reference
HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB': {
        'CMC': 1,
        'MCP': 2,
        'IP': 3,
        'TIP': 4
    },
    'INDEX': {
        'MCP': 5,
        'PIP': 6,
        'DIP': 7,
        'TIP': 8
    },
    'MIDDLE': {
        'MCP': 9,
        'PIP': 10,
        'DIP': 11,
        'TIP': 12
    },
    'RING': {
        'MCP': 13,
        'PIP': 14,
        'DIP': 15,
        'TIP': 16
    },
    'PINKY': {
        'MCP': 17,
        'PIP': 18,
        'DIP': 19,
        'TIP': 20
    }
}


class HandDetector:
    """A class for detecting and tracking hands using MediaPipe.
    
    This class provides methods for:
    - Detecting hands in images
    - Tracking hand landmarks
    - Drawing hand connections
    - Getting landmark positions
    """
    
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        """Initialize the hand detector.
        
        Args:
            mode (bool): Static mode for processing individual frames
            max_hands (int): Maximum number of hands to detect (1-2)
            detection_conf (float): Minimum detection confidence (0-1)
            track_conf (float): Minimum tracking confidence (0-1)
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
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(255, 0, 255),  # Pink color for landmarks
            thickness=2,
            circle_radius=4
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),  # Green color for connections
            thickness=2
        )
        
        # Initialize results storage
        self.results = None

    def find_hands(self, img, draw=True):
        """Detect and optionally draw hands in an image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format
            draw (bool): Whether to draw hand landmarks and connections
            
        Returns:
            numpy.ndarray: Processed image with or without drawings
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
            img (numpy.ndarray): Input image
            hand_no (int): Which hand to track (0 is first detected hand)
            draw (bool): Whether to draw position markers
            
        Returns:
            list: List of landmark positions [id, x, y] or empty list if no hand found
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            # Check if requested hand exists
            if hand_no < len(self.results.multi_hand_landmarks):
                target_hand = self.results.multi_hand_landmarks[hand_no]
                
                for idx, landmark in enumerate(target_hand.landmark):
                    height, width, _ = img.shape
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([idx, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                        # Optionally add landmark name for key points
                        if idx in [HAND_LANDMARKS['THUMB']['TIP'], 
                                 HAND_LANDMARKS['INDEX']['TIP']]:
                            cv2.putText(img, str(idx), (cx+10, cy+10),
                                      cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
                    
        return landmark_list


def main():
    """Main function demonstrating hand tracking capabilities."""
    
    # Initialize video capture and detector
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    detector = HandDetector(max_hands=2)
    prev_time = 0
    
    while True:
        # Read frame
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Process frame
        img = detector.find_hands(img)
        landmark_list = detector.find_positions(img)
        
        # Example: Print thumb tip position if hand is detected
        if landmark_list:
            thumb_tip = landmark_list[HAND_LANDMARKS['THUMB']['TIP']]
            print(f"Thumb tip position: {thumb_tip}")
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        # Show result
        cv2.imshow("Hand Tracking", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
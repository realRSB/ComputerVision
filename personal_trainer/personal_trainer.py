"""
Personal Trainer Script
-----------------------
This script provides real-time pose detection and angle measurement capabilities using MediaPipe.
It tracks various body landmarks, calculates angles between joints, and counts repetitions of exercises.

Key Features:
- Real-time pose detection
- Joint angle calculations
- Exercise repetition counting
- FPS monitoring
- Customizable visualization
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Define landmark indices for different body parts
POSE_LANDMARKS = {
    'RIGHT_ARM': {'SHOULDER': 12, 'ELBOW': 14, 'WRIST': 16},
    'LEFT_ARM': {'SHOULDER': 11, 'ELBOW': 13, 'WRIST': 15},
    'RIGHT_LEG': {'HIP': 24, 'KNEE': 26, 'ANKLE': 28},
    'LEFT_LEG': {'HIP': 23, 'KNEE': 25, 'ANKLE': 27}
}

class PoseDetector:
    """A class for detecting and tracking poses using MediaPipe."""
    
    def __init__(self, mode=False, up_body=False, smooth=True, detection_conf=0.5, track_conf=0.5):
        """Initialize the pose detector with custom parameters."""
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        # Initialize MediaPipe components
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        
        # Drawing styles
        self.landmark_style = self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        self.connection_style = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
        
        # Initialize results storage
        self.results = None
        self.landmark_list = []

    def find_pose(self, img, draw=True):
        """Process an image and detect pose landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.landmark_style,
                self.connection_style
            )
        return img

    def find_position(self, img, draw=True):
        """Extract landmark positions from the detected pose."""
        self.landmark_list = []
        
        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append([idx, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return self.landmark_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """Calculate and visualize the angle between three points."""
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = angle + 360 if angle < 0 else angle
        
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
            cv2.putText(img, f"{int(angle)} deg", (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return angle

def main():
    """Main function demonstrating pose estimation capabilities."""
    
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    count = 0
    direction = 0
    prev_time = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        img = cv2.resize(img, (1280, 720))
        img = detector.find_pose(img)
        landmark_list = detector.find_position(img)
        
        if len(landmark_list) != 0:
            # Right Arm
            angle = detector.find_angle(img, 11, 13, 15)
            percentage = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            
            color = (255, 0, 255)
            if percentage == 100:
                color = (0, 255, 0)
                if direction == 0:
                    count += 0.5
                    direction = 1
            if percentage == 0:
                color = (0, 255, 0)
                if direction == 1:
                    count += 0.5
                    direction = 0
            
            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(percentage)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
            
            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
        cv2.imshow("Pose Estimation", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
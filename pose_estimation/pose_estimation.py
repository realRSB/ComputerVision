"""
Pose Estimation Module
---------------------
This module provides real-time pose detection and angle measurement capabilities using MediaPipe.
It can track various body landmarks and calculate angles between joints.

Key Features:
- Real-time pose detection
- Joint angle calculations
- FPS monitoring
- Customizable visualization
"""

import cv2
import mediapipe as mp
import time
import math


# Define landmark indices for different body parts
POSE_LANDMARKS = {
    # Arm landmarks
    'RIGHT_ARM': {
        'SHOULDER': 12,
        'ELBOW': 14,
        'WRIST': 16
    },
    'LEFT_ARM': {
        'SHOULDER': 11,
        'ELBOW': 13,
        'WRIST': 15
    },
    # Leg landmarks
    'RIGHT_LEG': {
        'HIP': 24,
        'KNEE': 26,
        'ANKLE': 28
    },
    'LEFT_LEG': {
        'HIP': 23,
        'KNEE': 25,
        'ANKLE': 27
    }
}


class PoseDetector:
    """A class for detecting and tracking poses using MediaPipe.
    
    This class provides methods for:
    - Detecting pose landmarks
    - Calculating joint angles
    - Visualizing results
    """
    
    def __init__(self, mode=False, up_body=False, smooth=True, detection_conf=0.5, track_conf=0.5):
        """Initialize the pose detector with custom parameters.
        
        Args:
            mode (bool): Static mode for processing individual frames
            up_body (bool): Upper body only mode (False for full body)
            smooth (bool): Enable temporal landmark smoothing
            detection_conf (float): Minimum detection confidence (0-1)
            track_conf (float): Minimum tracking confidence (0-1)
        """
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
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=self.smooth,
            enable_segmentation=False,  # Set to True if you want segmentation
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        
        # Drawing styles
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(255, 0, 0),  # Red color for landmarks
            thickness=2,
            circle_radius=2
        )
        self.connection_style = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),  # Green color for connections
            thickness=2
        )
        
        # Initialize results storage
        self.results = None
        self.landmark_list = []

    def find_pose(self, img, draw=True):
        """Process an image and detect pose landmarks.
        
        Args:
            img (numpy.ndarray): Input image in BGR format
            draw (bool): Whether to draw landmarks on the image
            
        Returns:
            numpy.ndarray: Processed image with or without landmarks
        """
        # Convert BGR to RGB for MediaPipe
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
        """Extract landmark positions from the detected pose.
        
        Args:
            img (numpy.ndarray): Input image
            draw (bool): Whether to draw position markers
            
        Returns:
            list: List of landmark positions [id, x, y]
        """
        self.landmark_list = []
        
        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = img.shape
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append([idx, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return self.landmark_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        """Calculate and visualize the angle between three points.
        
        Args:
            img (numpy.ndarray): Input image
            p1 (int): First point index (start point)
            p2 (int): Second point index (middle/vertex point)
            p3 (int): Third point index (end point)
            draw (bool): Whether to visualize the angle
            
        Returns:
            float: Calculated angle in degrees
        """
        # Get landmark coordinates
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]
        
        # Calculate angle using arctangent
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                           math.atan2(y1 - y2, x1 - x2))
        
        # Ensure angle is positive (0-360 degrees)
        angle = angle + 360 if angle < 0 else angle
        
        if draw:
            # Draw connecting lines
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            
            # Draw points
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)  # Inner circle
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)  # Outer circle
            
            # Display angle near the vertex point
            cv2.putText(img, f"{int(angle)} deg", (x2 - 50, y2 + 50),
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        return angle

    def get_angle(self, img, p1, p2, p3):
        """Calculate angle between three points.
        
        Args:
            img: Input image
            p1: First point index (start)
            p2: Second point index (middle/vertex)
            p3: Third point index (end)
        
        Returns:
            float: Calculated angle in degrees
        """
        if not self.landmark_list:
            return 0
        
        # Get coordinates
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]
        
        # Calculate angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                            math.atan2(y1 - y2, x1 - x2))
        
        # Make angle positive
        if angle < 0:
            angle += 360
        
        return angle


def process_angles(detector, img, landmark_list):
    """Process and display joint angles if landmarks are detected.
    
    Args:
        detector (PoseDetector): Instance of PoseDetector
        img (numpy.ndarray): Input image
        landmark_list (list): List of detected landmarks
        
    Returns:
        dict: Dictionary of calculated angles
    """
    angles = {}
    
    if landmark_list and len(landmark_list) >= 33:
        try:
            # Process right arm
            landmarks = POSE_LANDMARKS['RIGHT_ARM']
            points = [landmarks['SHOULDER'], landmarks['ELBOW'], landmarks['WRIST']]
            if all(i < len(landmark_list) for i in points):
                angles['Right Arm'] = detector.find_angle(img, *points)
            
            # Process left arm
            landmarks = POSE_LANDMARKS['LEFT_ARM']
            points = [landmarks['SHOULDER'], landmarks['ELBOW'], landmarks['WRIST']]
            if all(i < len(landmark_list) for i in points):
                angles['Left Arm'] = detector.find_angle(img, *points)
            
            # Process right leg
            landmarks = POSE_LANDMARKS['RIGHT_LEG']
            points = [landmarks['HIP'], landmarks['KNEE'], landmarks['ANKLE']]
            if all(i < len(landmark_list) for i in points):
                angles['Right Leg'] = detector.find_angle(img, *points)
            
            # Process left leg
            landmarks = POSE_LANDMARKS['LEFT_LEG']
            points = [landmarks['HIP'], landmarks['KNEE'], landmarks['ANKLE']]
            if all(i < len(landmark_list) for i in points):
                angles['Left Leg'] = detector.find_angle(img, *points)
            
        except Exception as e:
            print(f"Error calculating angles: {e}")
            
    return angles


def display_info(img, angles, fps):
    """Display angles and FPS information on the image.
    
    Args:
        img (numpy.ndarray): Input image
        angles (dict): Dictionary of calculated angles
        fps (float): Current FPS value
    """
    # Display angles
    y_position = 30
    for joint, angle in angles.items():
        cv2.putText(img, f"{joint}: {int(angle)} deg", (10, y_position),
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        y_position += 30
    
    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, y_position),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


def main():
    """Main function demonstrating pose estimation capabilities."""
    
    # Initialize video capture and detector
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    detector = PoseDetector()
    prev_time = 0
    
    while True:
        # Read frame
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        # Process frame
        img = detector.find_pose(img)
        landmark_list = detector.find_position(img)
        
        # Calculate angles
        angles = process_angles(detector, img, landmark_list)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display information
        display_info(img, angles, fps)
        
        # Show result
        cv2.imshow("Pose Estimation", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
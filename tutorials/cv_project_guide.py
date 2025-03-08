"""
Computer Vision Project Guide
---------------------------
This guide demonstrates how to create your own computer vision project
using MediaPipe and OpenCV. It covers the basic structure and common patterns
used in projects like hand tracking, pose estimation, and face detection.

Key Concepts:
- Project structure
- MediaPipe integration
- Real-time processing
- Visualization
- Performance monitoring
"""

import cv2
import mediapipe as mp
import time


class VisionDetector:
    """Template class for creating vision detection projects.
    
    This class provides a basic structure that can be adapted for:
    - Hand tracking
    - Pose estimation
    - Face detection
    - Object detection
    etc.
    """
    
    def __init__(self, mode=False, complexity=1, detection_conf=0.5, track_conf=0.5):
        """Initialize your detector.
        
        Common parameters pattern:
        - Static/dynamic mode
        - Model complexity
        - Detection confidence
        - Tracking confidence
        """
        # Store configuration
        self.mode = mode
        self.complexity = complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        # Initialize MediaPipe components
        # Example for different solutions:
        """
        # For hands:
        self.solution = mp.solutions.hands
        self.detector = self.solution.Hands(
            static_image_mode=mode,
            max_num_hands=2,
            min_detection_confidence=detection_conf
        )
        
        # For pose:
        self.solution = mp.solutions.pose
        self.detector = self.solution.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            min_detection_confidence=detection_conf
        )
        
        # For face:
        self.solution = mp.solutions.face_detection
        self.detector = self.solution.FaceDetection(
            min_detection_confidence=detection_conf
        )
        """
        
        # Initialize drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        
        # Define drawing styles (customize colors, thickness)
        self.drawing_styles = {
            'landmark_style': self.mp_draw.DrawingSpec(
                color=(255, 0, 255),  # Magenta
                thickness=2,
                circle_radius=2
            ),
            'connection_style': self.mp_draw.DrawingSpec(
                color=(0, 255, 0),  # Green
                thickness=2
            )
        }
        
        # Initialize results storage
        self.results = None

    def process_image(self, img, draw=True):
        """Process a single image frame.
        
        Common steps:
        1. Convert color space (BGR to RGB)
        2. Process with MediaPipe
        3. Draw results if requested
        4. Return processed image
        """
        # Convert color space
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process image
        self.results = self.detector.process(img_rgb)
        
        # Draw results if requested
        if draw and self.results:
            self.draw_results(img)
            
        return img

    def draw_results(self, img):
        """Draw detection results on the image.
        
        Common patterns:
        - Draw landmarks
        - Draw connections
        - Add labels
        - Show confidence scores
        """
        # Example drawing code:
        """
        # For landmarks and connections:
        self.mp_draw.draw_landmarks(
            img,
            self.results.pose_landmarks,  # or hand_landmarks, face_landmarks
            self.solution.POSE_CONNECTIONS,  # or HAND_CONNECTIONS
            self.drawing_styles['landmark_style'],
            self.drawing_styles['connection_style']
        )
        
        # For bounding boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        # For labels:
        cv2.putText(img, "Label", (x, y), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        """
        pass


def create_demo():
    """Create a basic demo application.
    
    Common structure:
    1. Initialize camera and detector
    2. Process frames in real-time
    3. Display results
    4. Handle user input
    5. Cleanup resources
    """
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for webcam
    detector = VisionDetector()
    prev_time = 0
    
    while True:
        # Read frame
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        # Process frame
        img = detector.process_image(img)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        # Display result
        cv2.imshow("Vision Demo", img)
        
        # Handle 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function with usage examples."""
    print("Common steps to create your vision project:")
    print("1. Choose your MediaPipe solution (hands, pose, face, etc.)")
    print("2. Create a detector class inheriting from VisionDetector")
    print("3. Implement the process_image and draw_results methods")
    print("4. Create a demo application to test your detector")
    print("5. Add custom features (gesture recognition, tracking, etc.)")
    print("\nRunning basic demo...")
    
    create_demo()


if __name__ == "__main__":
    main() 
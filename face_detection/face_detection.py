"""
Face Detection Module
-------------------
This module provides real-time face detection capabilities using MediaPipe.
It can detect faces and draw fancy bounding boxes around them with confidence scores.

Key Features:
- Real-time face detection
- Customizable detection confidence
- Fancy bounding box visualization
- FPS monitoring
"""

import cv2
import mediapipe as mp
import time


class FaceDetector:
    """A class for detecting faces using MediaPipe.
    
    This class provides methods for:
    - Detecting faces in images
    - Drawing fancy bounding boxes
    - Displaying confidence scores
    """
    
    def __init__(self, min_detection_conf=0.5):
        """Initialize the face detector.
        
        Args:
            min_detection_conf (float): Minimum detection confidence threshold (0-1)
        """
        self.min_detection_conf = min_detection_conf
        
        # Initialize MediaPipe components
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_conf)
        
        # Initialize results storage
        self.results = None

    def find_faces(self, img, draw=True):
        """Detect faces in an image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format
            draw (bool): Whether to draw detection visualizations
            
        Returns:
            tuple: (Processed image, List of face detections [id, bbox, confidence])
        """
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        
        bboxes = []
        if self.results.detections:
            for idx, detection in enumerate(self.results.detections):
                # Get bounding box coordinates
                bbox_data = detection.location_data.relative_bounding_box
                height, width, _ = img.shape
                
                # Convert relative coordinates to absolute pixels
                bbox = (
                    int(bbox_data.xmin * width),
                    int(bbox_data.ymin * height),
                    int(bbox_data.width * width),
                    int(bbox_data.height * height)
                )
                
                bboxes.append([idx, bbox, detection.score])
                
                if draw:
                    # Draw fancy box and confidence score
                    img = self.fancy_draw(img, bbox)
                    confidence = int(detection.score[0] * 100)
                    cv2.putText(img, f'{confidence}%',
                              (bbox[0], bbox[1] - 20),
                              cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    
        return img, bboxes

    def fancy_draw(self, img, bbox, line_length=30, thickness=5, rectangle_thickness=1):
        """Draw a fancy bounding box with corner highlights.
        
        Args:
            img (numpy.ndarray): Input image
            bbox (tuple): Bounding box coordinates (x, y, w, h)
            line_length (int): Length of corner lines
            thickness (int): Thickness of corner lines
            rectangle_thickness (int): Thickness of main rectangle
            
        Returns:
            numpy.ndarray: Image with fancy bounding box
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        
        # Draw main rectangle
        cv2.rectangle(img, bbox, (255, 0, 255), rectangle_thickness)
        
        # Draw corner lines
        # Top Left
        cv2.line(img, (x, y), (x + line_length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y + line_length), (255, 0, 255), thickness)
        
        # Top Right
        cv2.line(img, (x1, y), (x1 - line_length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + line_length), (255, 0, 255), thickness)
        
        # Bottom Left
        cv2.line(img, (x, y1), (x + line_length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - line_length), (255, 0, 255), thickness)
        
        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - line_length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - line_length), (255, 0, 255), thickness)
        
        return img


def main():
    """Main function demonstrating face detection capabilities."""
    
    # Initialize video capture and detector
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    detector = FaceDetector(min_detection_conf=0.5)
    prev_time = 0
    
    while True:
        # Read frame
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        # Detect faces
        img, bboxes = detector.find_faces(img)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        # Show result
        cv2.imshow("Face Detection", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
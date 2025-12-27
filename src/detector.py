"""
Object detection module using YOLO models.
Supports YOLOv8, YOLOv5, and Roboflow models.
"""

import cv2
import numpy as np
from typing import List, Optional
from pathlib import Path
import supervision as sv
from ultralytics import YOLO


class VehicleDetector:
    """
    Vehicle detection using YOLO models.
    Detects and filters vehicles from video frames.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        iou: float = 0.7,
        device: str = "cuda",
        target_classes: Optional[List[int]] = None
    ):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to YOLO model weights (e.g., 'models/yolov8n.pt')
            confidence: Minimum confidence threshold for detections (0-1)
            iou: IOU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
            target_classes: List of COCO class IDs to detect (e.g., [2, 3, 5, 7])
                          Default is [2, 3, 5, 7] for car, motorcycle, bus, truck
        
        Note:
            ⚠️ Make sure the model file exists at model_path before initialization
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        
        # Default to common vehicle classes if not specified
        self.target_classes = target_classes or [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Load the YOLO model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model from disk."""
        if not self.model_path.exists():
            print(f"⚠️ Model not found at {self.model_path}")
            print("Attempting to download YOLOv8n model...")
            self.model = YOLO("yolov8n.pt")  # Will auto-download
        else:
            self.model = YOLO(str(self.model_path))
        
        # Move model to specified device
        self.model.to(self.device)
        print(f"✓ Model loaded successfully on {self.device}")
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            Supervision Detections object containing:
                - xyxy: Bounding boxes in [x1, y1, x2, y2] format
                - confidence: Detection confidence scores
                - class_id: Class IDs for each detection
                - tracker_id: (optional) Tracking IDs if tracking is enabled
        
        Example:
            >>> detector = VehicleDetector("models/yolov8n.pt")
            >>> frame = cv2.imread("traffic.jpg")
            >>> detections = detector.detect(frame)
            >>> print(f"Found {len(detections)} vehicles")
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.target_classes,  # Filter to vehicle classes only
            verbose=False  # Suppress YOLO output
        )[0]
        
        # Convert YOLO results to Supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """
        Detect vehicles in multiple frames (batch processing).
        
        Args:
            frames: List of input images
        
        Returns:
            List of Detections objects, one per frame
        
        Note:
            Batch processing can be faster for multiple frames but uses more memory
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
        
        return all_detections
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get the class name for a given class ID.
        
        Args:
            class_id: COCO class ID
        
        Returns:
            Human-readable class name
        """
        # COCO class names for vehicles
        class_names = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        return class_names.get(class_id, f"class_{class_id}")
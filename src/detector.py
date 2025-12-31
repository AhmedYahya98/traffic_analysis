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
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold
            iou: IOU threshold for NMS
            device: Device to run inference on (cuda/cpu/mps)
            target_classes: List of class indices to detect (from trained dataset)
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        
        # Default to dataset2 vehicle classes (articulated-bus, bus, car, motorbike, truck)
        self.target_classes = target_classes or [1, 2, 3, 5, 7]
        
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
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
        
        return all_detections
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get the class name for a given class ID.
        Uses dataset2 class mapping (8 classes).
        """
        # Dataset2 class names (from data/dataset2/data.yaml)
        class_names = {
            0: "PMT",
            1: "articulated-bus",
            2: "bus",
            3: "car",
            4: "freight",
            5: "motorbike",
            6: "small-bus",
            7: "truck"
        }
        return class_names.get(class_id, f"unknown_class_{class_id}")
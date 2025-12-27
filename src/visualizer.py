"""
Visualization module for annotating frames with detection results.
Creates visual overlays with bounding boxes, labels, zones, and statistics.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import supervision as sv


class TrafficVisualizer:
    """
    Visualize detection, tracking, and analysis results on video frames.
    Provides customizable annotations and overlays.
    """
    
    def __init__(
        self,
        class_names: Dict[int, str],
        show_boxes: bool = True,
        show_labels: bool = True,
        show_tracks: bool = True,
        show_zones: bool = True,
        thickness: int = 2,
        font_scale: float = 0.6
    ):
        """
        Initialize the visualizer.
        
        Args:
            class_names: Mapping of class IDs to names
            show_boxes: Whether to draw bounding boxes
            show_labels: Whether to show class labels
            show_tracks: Whether to show tracking IDs
            show_zones: Whether to draw counting zones
            thickness: Line thickness for boxes and zones
            font_scale: Text size for labels
        """
        self.class_names = class_names
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_tracks = show_tracks
        self.show_zones = show_zones
        self.thickness = thickness
        self.font_scale = font_scale
        
        # Initialize Supervision annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness,
            text_thickness=thickness,
            text_scale=font_scale
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_scale=font_scale,
            text_thickness=thickness
        )
        
        # Zone annotators (will be set when zones are provided)
        self.zone_annotators = []
    
    def set_zones(self, zones: List[Dict]):
        """
        Set up zone visualization.
        
        Args:
            zones: List of zone configurations with 'polygon' coordinates
        """
        self.zone_annotators = []
        for zone_config in zones:
            polygon = np.array(zone_config['polygon'], dtype=np.int32)
            zone = sv.PolygonZone(polygon=polygon)
            zone_annotator = sv.PolygonZoneAnnotator(
                zone=zone,
                color=sv.Color.from_hex("#FF0000"),
                thickness=self.thickness
            )
            self.zone_annotators.append({
                'name': zone_config['name'],
                'zone': zone,
                'annotator': zone_annotator
            })
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        statistics: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Annotate a frame with all visual elements.
        
        Args:
            frame: Input frame to annotate
            detections: Detections to visualize
            statistics: Optional statistics to display as overlay
        
        Returns:
            Annotated frame
        
        The annotated frame includes:
            - Bounding boxes around detected vehicles
            - Class labels and confidence scores
            - Tracking IDs (if tracking enabled)
            - Counting zones (if configured)
            - Statistics overlay (if provided)
        """
        annotated_frame = frame.copy()
        
        # Draw bounding boxes
        if self.show_boxes and len(detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )
        
        # Add labels
        if self.show_labels and len(detections) > 0:
            labels = self._create_labels(detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )
        
        # Draw zones
        if self.show_zones:
            for zone_obj in self.zone_annotators:
                annotated_frame = zone_obj['annotator'].annotate(
                    scene=annotated_frame
                )
        
        # Add statistics overlay
        if statistics:
            annotated_frame = self._draw_statistics(annotated_frame, statistics)
        
        return annotated_frame
    
    def _create_labels(self, detections: sv.Detections) -> List[str]:
        """
        Create label strings for each detection.
        
        Args:
            detections: Detections to create labels for
        
        Returns:
            List of label strings
        """
        labels = []
        
        for idx in range(len(detections)):
            class_id = detections.class_id[idx]
            confidence = detections.confidence[idx]
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Build label string
            label_parts = [f"{class_name} {confidence:.2f}"]
            
            # Add tracking ID if available
            if (self.show_tracks and hasattr(detections, 'tracker_id') and 
                detections.tracker_id is not None):
                tracker_id = detections.tracker_id[idx]
                label_parts.append(f"ID:{tracker_id}")
            
            labels.append(" | ".join(label_parts))
        
        return labels
    
    def _draw_statistics(
        self,
        frame: np.ndarray,
        statistics: Dict
    ) -> np.ndarray:
        """
        Draw statistics overlay on frame.
        
        Args:
            frame: Frame to draw on
            statistics: Statistics dictionary from analyzer
        
        Returns:
            Frame with statistics overlay
        """
        # Create semi-transparent overlay box
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Statistics box dimensions
        box_width = 300
        box_height = 150
        box_x = width - box_width - 20
        box_y = 20
        
        # Draw background rectangle
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 0, 0),
            -1
        )
        
        # Blend overlay with frame
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add text
        y_offset = box_y + 30
        text_color = (255, 255, 255)
        
        # Total vehicles
        cv2.putText(
            frame,
            f"Total: {statistics.get('unique_vehicles', 0)}",
            (box_x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )
        y_offset += 30
        
        # Per-class counts
        unique_per_class = statistics.get('unique_per_class', {})
        for class_name, count in unique_per_class.items():
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (box_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
            y_offset += 25
        
        return frame
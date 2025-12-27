"""
Traffic analysis module for counting and categorizing vehicles.
Implements zone-based counting and statistics collection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import supervision as sv


class TrafficAnalyzer:
    """
    Analyze traffic patterns, count vehicles, and collect statistics.
    Supports zone-based counting for entry/exit analysis.
    """
    
    def __init__(
        self,
        class_names: Dict[int, str],
        zones: Optional[List[Dict]] = None
    ):
        """
        Initialize the traffic analyzer.
        
        Args:
            class_names: Mapping of class IDs to human-readable names
                        Example: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
            zones: List of zone definitions for counting
                  Example: [{"name": "entry", "polygon": [[x1,y1], [x2,y2], ...]}]
        
        The analyzer tracks:
            - Total count per class
            - Unique vehicles seen (by tracker ID)
            - Per-zone counts (vehicles entering/exiting zones)
        """
        self.class_names = class_names
        self.zones = zones or []
        
        # Statistics storage
        self.total_counts = defaultdict(int)  # Count per class
        self.unique_ids = set()  # All unique tracker IDs seen
        self.class_wise_ids = defaultdict(set)  # IDs per class
        self.zone_counts = defaultdict(lambda: defaultdict(int))  # Per-zone counts
        
        # Initialize zone objects if zones are provided
        self.zone_objects = []
        if self.zones:
            self._initialize_zones()
    
    def _initialize_zones(self):
        """Initialize Supervision PolygonZone objects from zone definitions."""
        for zone_config in self.zones:
            polygon = np.array(zone_config['polygon'], dtype=np.int32)
            zone = sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=(1920, 1080)  # Adjust based on your video
            )
            self.zone_objects.append({
                'name': zone_config['name'],
                'zone': zone,
                'counter': defaultdict(int)
            })
        print(f"✓ Initialized {len(self.zone_objects)} counting zones")
    
    def update(self, detections: sv.Detections, frame_shape: Tuple[int, int]):
        """
        Update statistics with detections from current frame.
        
        Args:
            detections: Tracked detections from current frame
            frame_shape: (height, width) of the frame for zone resolution
        
        This method:
            1. Counts total detections per class
            2. Tracks unique vehicle IDs
            3. Updates zone-based counts
        """
        # Update total counts per class
        for class_id in detections.class_id:
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            self.total_counts[class_name] += 1
        
        # Track unique vehicles (requires tracking to be enabled)
        if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                self.unique_ids.add(tracker_id)
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                self.class_wise_ids[class_name].add(tracker_id)
        
        # Update zone counts
        for zone_obj in self.zone_objects:
            zone = zone_obj['zone']
            # Trigger zone counting
            mask = zone.trigger(detections)
            
            # Count vehicles in zone by class
            for idx, in_zone in enumerate(mask):
                if in_zone:
                    class_id = detections.class_id[idx]
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    zone_obj['counter'][class_name] += 1
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive traffic statistics.
        
        Returns:
            Dictionary containing:
                - total_detections: Total number of detections
                - unique_vehicles: Number of unique vehicles tracked
                - class_counts: Counts per vehicle class
                - unique_per_class: Unique vehicles per class
                - zone_statistics: Per-zone counts (if zones configured)
        
        Example:
            >>> analyzer = TrafficAnalyzer(class_names={2: "car", 3: "motorcycle"})
            >>> # ... process frames ...
            >>> stats = analyzer.get_statistics()
            >>> print(f"Total vehicles: {stats['unique_vehicles']}")
        """
        stats = {
            'total_detections': sum(self.total_counts.values()),
            'unique_vehicles': len(self.unique_ids),
            'class_counts': dict(self.total_counts),
            'unique_per_class': {
                class_name: len(ids) 
                for class_name, ids in self.class_wise_ids.items()
            },
            'zone_statistics': {}
        }
        
        # Add zone statistics
        for zone_obj in self.zone_objects:
            zone_name = zone_obj['name']
            stats['zone_statistics'][zone_name] = dict(zone_obj['counter'])
        
        return stats
    
    def reset(self):
        """Reset all statistics. Useful when processing a new video."""
        self.total_counts = defaultdict(int)
        self.unique_ids = set()
        self.class_wise_ids = defaultdict(set)
        self.zone_counts = defaultdict(lambda: defaultdict(int))
        
        # Reset zone counters
        for zone_obj in self.zone_objects:
            zone_obj['counter'] = defaultdict(int)
        
        print("✓ Analyzer statistics reset")
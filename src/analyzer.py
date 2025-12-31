import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import supervision as sv


class TrafficAnalyzer:
    def __init__(
        self,
        class_names: Dict[int, str],
        zones: Optional[List[Dict]] = None
    ):
        """
        Initialize the traffic analyzer.
        
        Args:
            class_names: Mapping of class IDs to human-readable names.
                        Must match dataset class indices (from config.yaml).
                        Example: {0: "PMT", 1: "articulated-bus", 2: "bus", 3: "car", 4: "freight", 5: "motorbike", 6: "small-bus", 7: "truck"}
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
                polygon=polygon
            )

            self.zone_objects.append({
                'name': zone_config['name'],
                'zone': zone,
                'counter': defaultdict(int),
                'tracker_in_zone': set()  # Track which vehicles have entered this zone
            })

        print(f"✓ Initialized {len(self.zone_objects)} counting zones")
    
    def update(self, detections: sv.Detections, frame_shape: Tuple[int, int]):
        """
        Update statistics with detections from current frame.
        
        Args:
            detections: Tracked detections from current frame
            frame_shape: (height, width) of the frame for zone resolution
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
            
            # Count vehicles in zone by class (only count once per unique vehicle)
            for idx, in_zone in enumerate(mask):
                if in_zone:
                    tracker_id = None
                    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                        tracker_id = detections.tracker_id[idx]
                    
                    # Only count if this is a new entry to the zone
                    if tracker_id is not None:
                        if tracker_id not in zone_obj['tracker_in_zone']:
                            class_id = detections.class_id[idx]
                            class_name = self.class_names.get(class_id, f"unknown_class_{class_id}")
                            zone_obj['counter'][class_name] += 1
                            zone_obj['tracker_in_zone'].add(tracker_id)
                    else:
                        # Fallback if tracking is disabled (count every detection)
                        class_id = detections.class_id[idx]
                        class_name = self.class_names.get(class_id, f"unknown_class_{class_id}")
                        zone_obj['counter'][class_name] += 1
            
            # Remove tracking IDs that are no longer in the zone
            current_ids = set()
            for idx, in_zone in enumerate(mask):
                if in_zone and hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    current_ids.add(detections.tracker_id[idx])
            zone_obj['tracker_in_zone'] &= current_ids  # Keep only IDs still in zone
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive traffic statistics.        
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
        
        # Reset zone counters and tracking state
        for zone_obj in self.zone_objects:
            zone_obj['counter'] = defaultdict(int)
            zone_obj['tracker_in_zone'] = set()
        
        print("✓ Analyzer statistics reset")
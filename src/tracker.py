"""
Multi-object tracking module using ByteTrack.
Maintains consistent IDs for detected objects across frames.
"""

import numpy as np
from typing import Optional
import supervision as sv


class VehicleTracker:
    """
    Track vehicles across video frames using ByteTrack algorithm.
    Assigns and maintains unique IDs for each vehicle.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        """
        Initialize the vehicle tracker.
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks alive
            match_thresh: IOU threshold for matching detections to tracks
            frame_rate: Video frame rate (for track buffer calculation)
        
        How ByteTrack works:
            1. Associates high-confidence detections with existing tracks
            2. Uses low-confidence detections to recover lost tracks
            3. Removes tracks that haven't been updated for track_buffer frames
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        # Initialize ByteTrack tracker from Supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        
        print(f"✓ Tracker initialized with ByteTrack algorithm")
    
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracks with new detections from current frame.
        
        Args:
            detections: Detections from current frame (from detector)
        
        Returns:
            Detections with tracker_id field added
        
        Note:
            The returned detections will have a 'tracker_id' field containing
            unique IDs for each tracked object. IDs persist across frames.
        
        Example:
            >>> tracker = VehicleTracker()
            >>> detections = detector.detect(frame)
            >>> tracked_detections = tracker.update(detections)
            >>> for det_id in tracked_detections.tracker_id:
            >>>     print(f"Vehicle ID: {det_id}")
        """
        # Update tracker with new detections
        tracked_detections = self.tracker.update_with_detections(detections)
        
        return tracked_detections
    
    def reset(self):
        """
        Reset the tracker state.
        Useful when switching to a new video or restarting analysis.
        """
        self.tracker = sv.ByteTrack(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate
        )
        print("✓ Tracker reset")
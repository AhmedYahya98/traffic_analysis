"""
Export module for saving analysis results to various formats.
Supports video, JSON, and CSV outputs.
"""

import cv2
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class ResultsExporter:
    """
    Export traffic analysis results to files.
    Handles video, JSON, and CSV output formats.
    """
    
    def __init__(self, output_config: Dict):
        """
        InitializeArgs:
        output_config: Configuration dictionary with output paths and settings
        """
        self.output_config = output_config
        self.video_writer = None
        self.frame_results = []  # Store per-frame data for JSON export
        
        # Create output directories
        self._create_output_dirs()

    def _create_output_dirs(self):
        """Create output directories if they don't exist."""
        paths = [
            self.output_config.get('video_path'),
            self.output_config.get('json_path'),
            self.output_config.get('csv_path')
        ]
        
        for path in paths:
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)

    def initialize_video_writer(
        self,
        frame_width: int,
        frame_height: int,
        fps: int = 30
    ):
        """
        Initialize video writer for saving annotated video.
        
        Args:
            frame_width: Width of output video
            frame_height: Height of output video
            fps: Frames per second for output video
        
        Note:
            Call this before writing any frames
        """
        if not self.output_config.get('save_video', False):
            return
        
        video_path = self.output_config['video_path']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        print(f"✓ Video writer initialized: {video_path}")

    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to the output video.
        
        Args:
            frame: Annotated frame to write
        """
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def add_frame_data(
        self,
        frame_number: int,
        detections: Any,
        statistics: Dict
    ):
        """
        Store frame data for JSON export.
        
        Args:
            frame_number: Current frame index
            detections: Detections from this frame
            statistics: Current statistics
        """
        if not self.output_config.get('save_json', False):
            return
        
        # Convert detections to serializable format
        frame_data = {
            'frame_number': frame_number,
            'num_detections': len(detections),
            'detections': []
        }
        
        # Add each detection
        for idx in range(len(detections)):
            det_data = {
                'bbox': detections.xyxy[idx].tolist(),
                'confidence': float(detections.confidence[idx]),
                'class_id': int(detections.class_id[idx])
            }
            
            # Add tracker ID if available
            if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                det_data['tracker_id'] = int(detections.tracker_id[idx])
            
            frame_data['detections'].append(det_data)
        
        self.frame_results.append(frame_data)

    def save_json(self, final_statistics: Dict):
        """
        Save all results to JSON file.
        
        Args:
            final_statistics: Final statistics from analyzer
        """
        if not self.output_config.get('save_json', False):
            return
        
        output_data = {
            'summary': final_statistics,
            'frames': self.frame_results
        }
        
        json_path = self.output_config['json_path']
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ JSON results saved: {json_path}")

    def save_csv(self, statistics: Dict):
        """
        Save statistics to CSV file.
        
        Args:
            statistics: Statistics dictionary from analyzer
        """
        if not self.output_config.get('save_csv', False):
            return
        
        # Create DataFrame with statistics
        rows = []
        
        # Overall statistics
        rows.append({
            'metric': 'total_detections',
            'value': statistics.get('total_detections', 0)
        })
        rows.append({
            'metric': 'unique_vehicles',
            'value': statistics.get('unique_vehicles', 0)
        })
        
        # Per-class counts
        for class_name, count in statistics.get('class_counts', {}).items():
            rows.append({
                'metric': f'total_{class_name}',
                'value': count
            })
        
        # Unique per class
        for class_name, count in statistics.get('unique_per_class', {}).items():
            rows.append({
                'metric': f'unique_{class_name}',
                'value': count
            })
        
        # Zone statistics
        for zone_name, zone_stats in statistics.get('zone_statistics', {}).items():
            for class_name, count in zone_stats.items():
                rows.append({
                    'metric': f'{zone_name}_{class_name}',
                    'value': count
                })
        
        df = pd.DataFrame(rows)
        csv_path = self.output_config['csv_path']
        df.to_csv(csv_path, index=False)
        
        print(f"✓ CSV results saved: {csv_path}")

    def close(self):
        """Clean up resources (close video writer, etc.)."""
        if self.video_writer is not None:
            self.video_writer.release()
            print("✓ Video writer closed")

"""
Main application entry point for traffic analysis.
Orchestrates all components to process video and generate results.
"""

import cv2
import sys
from pathlib import Path
from tqdm import tqdm

# Import project modules
from src.config_loader import ConfigLoader
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.analyzer import TrafficAnalyzer
from src.visualizer import TrafficVisualizer
from src.exporter import ResultsExporter


def process_video(config: ConfigLoader):
    """
    Process a video file and generate traffic analysis results.
    
    Args:
        config: Loaded configuration object
    
    This function:
        1. Loads the input video
        2. Initializes all processing components
        3. Processes each frame (detect -> track -> analyze -> visualize)
        4. Exports results
    """
    # Get configuration sections
    model_config = config.get_model_config()
    input_config = config.get_input_config()
    tracking_config = config.get_tracking_config()
    analysis_config = config.get_analysis_config()
    viz_config = config.get_visualization_config()
    output_config = config.get_output_config()
    
    # ⚠️ Validate input video path
    video_path = input_config['video_path']
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        print("⚠️ Please update the video_path in config/config.yaml")
        sys.exit(1)
    
    # Initialize video capture
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file")
        sys.exit(1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
    # Initialize components
    print("\n=== Initializing Components ===")
    
    # 1. Detector
    detector = VehicleDetector(
        model_path=model_config['weights_path'],
        confidence=model_config['confidence'],
        iou=model_config['iou'],
        device=model_config['device'],
        target_classes=analysis_config['target_classes']
    )
    
    # 2. Tracker (if enabled)
    tracker = None
    if tracking_config['enabled']:
        tracker = VehicleTracker(
            track_thresh=tracking_config['track_thresh'],
            track_buffer=tracking_config['track_buffer'],
            match_thresh=tracking_config['match_thresh'],
            frame_rate=fps
        )
    
    # 3. Analyzer
    analyzer = TrafficAnalyzer(
        class_names=analysis_config['class_names'],
        zones=analysis_config.get('zones', [])
    )
    
    # 4. Visualizer
    visualizer = TrafficVisualizer(
        class_names=analysis_config['class_names'],
        show_boxes=viz_config.get('show_boxes', True),
        show_labels=viz_config.get('show_labels', True),
        show_tracks=viz_config.get('show_tracks', True),
        show_zones=viz_config.get('show_zones', True),
        thickness=viz_config.get('thickness', 2),
        font_scale=viz_config.get('font_scale', 0.6)
    )
    
    # Set up zones for visualization
    if analysis_config.get('zones'):
        visualizer.set_zones(analysis_config['zones'])
    
    # 5. Exporter
    exporter = ResultsExporter(output_config)
    exporter.initialize_video_writer(frame_width, frame_height, fps)
    
    print("\n=== Processing Video ===")
    
    # Process each frame
    frame_number = 0
    progress_bar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Detect vehicles
        detections = detector.detect(frame)
        
        # Step 2: Track vehicles (if enabled)
        if tracker is not None:
            detections = tracker.update(detections)
        
        # Step 3: Analyze traffic
        analyzer.update(detections, (frame_height, frame_width))
        
        # Step 4: Get current statistics
        statistics = analyzer.get_statistics()
        
        # Step 5: Visualize
        annotated_frame = visualizer.annotate_frame(frame, detections, statistics)
        
        # Step 6: Export
        exporter.write_frame(annotated_frame)
        exporter.add_frame_data(frame_number, detections, statistics)
        
        frame_number += 1
        progress_bar.update(1)
    
    progress_bar.close()
    cap.release()
    
    # Final export
    print("\n=== Exporting Results ===")
    final_statistics = analyzer.get_statistics()
    exporter.save_json(final_statistics)
    exporter.save_csv(final_statistics)
    exporter.close()
    
    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Total detections: {final_statistics['total_detections']}")
    print(f"Unique vehicles: {final_statistics['unique_vehicles']}")
    print("\nVehicles by type:")
    for class_name, count in final_statistics['unique_per_class'].items():
        print(f"  {class_name}: {count}")
    
    print(f"\n✓ Results saved to data/output/")


def main():
    """Main entry point."""
    print("="*60)
    print("Traffic Analysis System")
    print("="*60)
    
    # Load configuration
    try:
        config = ConfigLoader("config/config.yaml")
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Process video
    try:
        process_video(config)
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
            
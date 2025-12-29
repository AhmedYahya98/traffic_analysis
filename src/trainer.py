"""
Simple YOLO11n model trainer for traffic detection.
No complexity - just train!
"""

from ultralytics import YOLO
from pathlib import Path


def train_yolo11n(
    dataset_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cpu"
):
    """
    Train YOLO11n model on your dataset - super simple!
    
    Args:
        dataset_path: Path to your dataset folder
        epochs: How many times to go through the data (default: 50)
        batch_size: How many images per batch (default: 16, reduce if out of memory)
        device: "cpu" or "cuda" for GPU
    
    Your dataset should look like this:
        dataset_path/
        ├── train/
        │   ├── images/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
        │   └── labels/
        │       ├── img1.txt
        │       └── img2.txt
        └── val/
            ├── images/
            └── labels/
    
    Label format (in .txt files):
        class_id x_center y_center width height
        Example: 0 0.5 0.5 0.3 0.4
        (all values between 0 and 1)
    """
    
    print("="*60)
    print("Training YOLO11n Model")
    print("="*60)
    
    # Create data.yaml file
    dataset_path = Path(dataset_path)
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        print(f"\n⚠️ Creating data.yaml file...")
        
        # Try to detect classes from first label file
        label_files = list((dataset_path / "train" / "labels").glob("*.txt"))
        if label_files:
            with open(label_files[0], 'r') as f:
                max_class = max([int(line.split()[0]) for line in f if line.strip()])
            num_classes = max_class + 1
        else:
            num_classes = 4  # Default: car, motorcycle, bus, truck
        
        # Create simple data.yaml
        yaml_content = f"""path: {dataset_path.absolute()}
train: train/images
val: val/images

nc: {num_classes}
names: {list(range(num_classes))}
"""
        with open(data_yaml, 'w') as f:
            f.write(yaml_content)
        print(f"✓ Created {data_yaml}")
    
    # Load YOLO11n model
    print("\nLoading YOLO11n model...")
    model = YOLO("yolo11n.pt")  # Will auto-download if needed
    print("✓ Model loaded")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print("\n" + "="*60)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device,
        project="trained_models",
        name="yolo11n_traffic",
        patience=10,  # Stop early if no improvement
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    
    best_model = Path("trained_models/yolo11n_traffic/weights/best.pt")
    print(f"\n📦 Your trained model: {best_model}")
    print(f"\nTo use it, update config/config.yaml:")
    print(f'  weights_path: "{best_model}"')
    
    return str(best_model)


def validate_model(model_path: str, dataset_path: str, device: str = "cpu"):
    """
    Test your trained model.
    
    Args:
        model_path: Path to your trained model (best.pt)
        dataset_path: Path to dataset folder
        device: "cpu" or "cuda"
    """
    print("\n" + "="*60)
    print("Validating Model")
    print("="*60)
    
    model = YOLO(model_path)
    data_yaml = Path(dataset_path) / "data.yaml"
    
    results = model.val(data=str(data_yaml), device=device)
    
    print("\n📊 Results:")
    print(f"  mAP50: {results.box.map50:.3f}")
    print(f"  mAP50-95: {results.box.map:.3f}")
    print(f"  Precision: {results.box.mp:.3f}")
    print(f"  Recall: {results.box.mr:.3f}")
    
    return results

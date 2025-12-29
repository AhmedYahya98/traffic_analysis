from src.trainer import train_yolo11n, validate_model


# Simple training script
if __name__ == "__main__":
    
    DATASET_PATH = "C:\\Users\\roaay\\OneDrive\\Desktop\\traffic_analysis\\data\\my_dataset"  # Your dataset folder
    EPOCHS = 50                        # Training epochs
    BATCH_SIZE = 2                    # Images per batch (reduce if out of memory)
    DEVICE = "cuda"                     # "cpu" or "cuda"
    
    # Train
    print("Starting simple YOLO11n training...")
    trained_model = train_yolo11n(
        dataset_path=DATASET_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    # Validate
    print("\nTesting the trained model...")
    validate_model(
        model_path=trained_model,
        dataset_path=DATASET_PATH,
        device=DEVICE
    )
    
    print("\n✅ All done! Use your model by updating config/config.yaml")
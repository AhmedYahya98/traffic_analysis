from src.trainer import train_yolo, validate_model


# Simple training script
if __name__ == "__main__":
    
    DATASET_PATH = ""  # Your dataset folder
    EPOCHS = 10                        # Training epochs
    BATCH_SIZE = 2                    # Images per batch (reduce if out of memory)
    DEVICE = "cuda"                     # "cpu" or "cuda"
    
    # Train
    print("Starting simple YOLO training...")
    trained_model = train_yolo(
        dataset_path=DATASET_PATH,
        model="yolo11n.pt",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        patience=5
    )
    print("\nTraining complete!")
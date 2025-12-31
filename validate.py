from src.trainer import validate_model


# Simple training script
if __name__ == "__main__":
    
    DATASET_PATH = ""  # Your dataset folder
    DEVICE = "cuda"                     # "cpu" or "cuda"
    MODEL_PATH = ""  # Your trained model path (best.pt)
    # Validate
    print("\nTesting the trained model...")
    validate_model(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        device=DEVICE
    )
    
    print("\n✅ All done! Use your model by updating config/config.yaml")
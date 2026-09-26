import argparse
from pathlib import Path
from ultralytics import YOLO

def train_model(data_yaml: str, epochs: int = 100, batch_size: int = 16, model_name: str = "yolov8n.pt"):
    """
    Trains a YOLOv8 model on the hazard dataset.
    """
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Error: Data configuration file '{data_yaml}' not found.")
        return

    print(f"Initializing YOLOv8 model from {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return

    print(f"Starting training on dataset {data_yaml} for {epochs} epochs...")
    try:
        results = model.train(
            data=str(data_path.absolute()),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project="runs/detect",
            name="hazard_model",
            exist_ok=True,
            device="cpu"  # Change to "0" or "cuda" if GPU is available
        )
        print("Training completed successfully.")
        
        # Validate the trained model
        print("Running validation...")
        metrics = model.val()
        print(f"Validation map50-95: {metrics.box.map}")
        
    except Exception as e:
        print(f"Error during training or validation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionAssist Hazard Detection Model")
    parser.add_argument("--data", type=str, default="../models/hazard_dataset.yaml", help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="../yolov8n.pt", help="Base model to start from")
    
    args = parser.parse_args()
    
    # Adjust paths relative to the script location
    script_dir = Path(__file__).parent
    data_path = (script_dir / args.data).resolve()
    model_path = (script_dir / args.model).resolve()
    
    train_model(str(data_path), epochs=args.epochs, batch_size=args.batch, model_name=str(model_path))
